
# coding: utf-8

# #### Testing on subset of imdb data

from timeit import default_timer as timer
import html
import torch
import sys

import matplotlib.pylab as plt

from fastai.text import *

print('Active CUDA Device: GPU', torch.cuda.current_device())

BOS = 'xbos'    #beginning of sentence tag, useful for model to know this
FLD = 'xfld'    #data field tag

PATH=Path('..')/'data/imdb/aclImdb'

os.listdir(PATH)

#Classifier
CLAS_PATH=Path('..')/'data/imdb/imdb_clas'
CLAS_PATH.mkdir(exist_ok=True)

#Language Model
LM_PATH=Path('..')/'data/imdb/imdb_lm'
LM_PATH.mkdir(exist_ok=True)

#unsup for unlabelled
CLASSES = ['neg', 'pos', 'unsup']

DATA_SUBSET = 50 #1/50th of data
DATA_SUBSET_STR = '_' + str(DATA_SUBSET)

#this makes pandas more efficient-when passed in to pandas, returns an iterator to iterate through chunks, then loop through these chinks of the dataframe
CHUNKSIZE = 24000

#Compile a regular expression pattern, returning a pattern object
re1 = re.compile(r'  +')

#limit as over this code gets 'clunky'
MAX_VOCAB = 60000
MIN_FREQ = 2

#we cant change these as required to match wikitext103
# number of hidden activation per LSTM layer
n_hid = 1150
# number of LSTM layers to use in the architecture
n_layers = 3
EMBEDDING_SIZE = 400

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

#tweak these to determine effects
wd = 1e-7
# grab 70 at a time
bptt = 70
#bs = 52

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        #The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
        for fname in (path/label).glob('*.*'):
            #eg ../data/imdb/aclImdb/train/neg/1696_1.txt
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts), np.array(labels)

def create_class_files(data_subset_frac):
    trn_texts, trn_labels = get_texts(PATH/'train')
    val_texts, val_labels = get_texts(PATH/'test')
    print(f'len(trn_texts): {len(trn_texts)}, len(val_texts): {len(val_texts)}')

    #clip data
    trn_upper_clip = int(len(trn_texts)/data_subset_frac)
    val_upper_clip = int(len(val_texts) / data_subset_frac)

    trn_texts = trn_texts[:trn_upper_clip]
    trn_labels = trn_labels[:trn_upper_clip]
    val_texts = val_texts[:val_upper_clip]
    val_labels = val_labels[:val_upper_clip]

    print(f'clipped: len(trn_texts): {len(trn_texts)}, len(val_texts): {len(val_texts)}')

    col_names = ['labels', 'text']

    #make randomness reproducible
    np.random.seed(42)
    #randomly shuffle this list
    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))

    #create our randomly sorted training and validation lists-generally a good idea to do this
    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]

    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]

    df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

    #for training data, remove unsupervised
    df_trn[df_trn['labels']!=2].to_csv(str(CLAS_PATH)+'train'+'_' + str(data_subset_frac)+'.csv',header=False, index=False)
    df_val.to_csv(str(CLAS_PATH)+'test'+'_' + str(data_subset_frac)+'.csv', header=False, index=False)

    #write the classes to a file ie neg pos unsup
    clas_path = 'classes'+'_' + str(data_subset_frac)+'.txt'
    (CLAS_PATH/'{0}'.format(clas_path)).open('w').writelines(f'{o}/n' for o in CLASSES)
    return trn_texts, val_texts, col_names


def create_train_test_files(trn_texts, val_texts, col_names, data_subset_frac):

    #use more data for training than the given split
    trn_texts,val_texts = sklearn.model_selection.train_test_split(np.concatenate([trn_texts, val_texts]), test_size=0.1)

    len(trn_texts), len(val_texts)

    #initialise classifications to zero
    df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

    df_trn.to_csv(str(LM_PATH)+'/train'+'_' + str(data_subset_frac)+'.csv', header=False, index=False)
    df_val.to_csv(str(LM_PATH)+'/test'+'_' + str(data_subset_frac)+'.csv', header=False, index=False)


#this may not catch all badly formatted text, may need to add to/modify for other input datasets
def fixup(text_str):
    text_str = text_str.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(text_str))


def lookup_texts(df, n_lbls=1):
    #.iloc[<row_selection>,<col_selction>] here default is column 0 only
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 '+ df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)):
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)
    #type(texts): <class 'pandas.core.series.Series'>
    #uses ProcessPoolExcutor with 1/2 of the cpu's, pass in a series to tokenize
    start = timer()
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    end = timer()
    print(f'elapsed: {end - start}')
    return tok, list(labels)

def get_all(tf_reader, n_lbls):
    #iterate over the TextFileReader object in chunks
    tok, labels = [], []
    for i, r in enumerate(tf_reader):
        print(i)
        tok_, labels_ = lookup_texts(r, n_lbls)
        tok += tok_
        labels += labels_
    return tok, labels

def create_tokens(data_subset_frac):
    # ## Language Model Tokens
    #
    # Turn text into a a list of tokens using Spacy

    print(f'max thread count: {len(os.sched_getaffinity(0))}')

    df_trn = pd.read_csv(str(LM_PATH)+'/train'+'_' + str(data_subset_frac)+'.csv', header=None, chunksize=CHUNKSIZE)
    df_val = pd.read_csv(str(LM_PATH)+'/test'+'_' + str(data_subset_frac)+'.csv', header=None, chunksize=CHUNKSIZE)
    #note is not a dataframe
    print(f'type(df_trn): {type(df_trn)}')

    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)

    (LM_PATH/'tmp').mkdir(exist_ok=True)

    np.save(str(LM_PATH)+'/tmp/tok_trn'+'_' + str(data_subset_frac)+'.npy', tok_trn)
    np.save(str(LM_PATH)+'/tmp/tok_val'+'_' + str(data_subset_frac)+'.npy', tok_val)
    return tok_trn, tok_val


def create_token_lookups(tok_trn, tok_val, data_subset_frac):
    freq = Counter(p for o in tok_trn for p in o)
    freq.most_common(25)

    # itos: index to string
    # stoi: string to index

    #index those tokens that appear more than 2x
    itos = [o for o,c in freq.most_common(MAX_VOCAB) if c>MIN_FREQ]
    itos.insert(0, '_pad_')
    #use if not in vocab
    itos.insert(0, '_unk_')
    itos[2]

    #default to 0 if not in dict
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(stoi)

    #index each token for each review
    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
    trn_lm.shape

    np.save(str(LM_PATH)+'/tmp/trn_ids'+'_' + str(data_subset_frac)+'.npy', trn_lm)
    np.save(str(LM_PATH)+'/tmp/val_ids'+'_' + str(data_subset_frac)+'.npy', val_lm)
    pickle.dump(itos, open(str(LM_PATH)+'/tmp/itos'+'_' + str(data_subset_frac)+'.pkl','wb'))

def get_lookups(data_subset_frac):
    print('>>get_lookups()')
    trn_lm = np.load(str(LM_PATH)+'/tmp/trn_ids'+'_' + str(data_subset_frac)+'.npy')
    val_lm = np.load(str(LM_PATH)+'/tmp/val_ids'+'_' + str(data_subset_frac)+'.npy')
    itos = pickle.load(open(str(LM_PATH)+'/tmp/itos'+'_' + str(data_subset_frac)+'.pkl', 'rb'))
    return trn_lm, val_lm, itos

def wikitext103_conversion(itos, trn_lm, data_subset_frac):
    print('>>wikitext103_conversion()')
    VOCAB_SIZE = len(itos)
    VOCAB_SIZE, len(trn_lm)

    PRE_PATH = PATH/'models'/'wt103'
    PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

    #torch.load uses Python's unpickling facilities but treats storages, which underlie tensors, specially
    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)

    itos_wiki = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
    stoi_wiki = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos_wiki)})

    #we need to map our itos to itos for wikitext
    new_wgts = np.zeros((VOCAB_SIZE, EMBEDDING_SIZE), dtype=np.float32)

    for i, w in enumerate(itos):
        r = stoi_wiki[w]
        # use mean if our string doesnt exist in wiki
        new_wgts[i] = enc_wgts[r] if r >= 0 else row_m

    #T: convert to torch tensor and put on gpu
    wgts['0.encoder.weight'] = T(new_wgts)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_wgts))
    #decoder uses same weights
    wgts['1.decoder.weight'] = T(np.copy(new_wgts))
    pickle.dump(wgts, open(str(PRE_PATH)+'/wgts_wt103'+'_' + str(data_subset_frac)+'.pkl','wb'))
    return wgts


def language_model(trn_lm, val_lm, vocab_size, bs=52):
    print('>>language_model()')
    # ## Language Model

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)

    model_data = LanguageModelData(PATH, pad_idx=1, nt=vocab_size, trn_dl=trn_dl, val_dl=val_dl, bs=bs, bptt=bptt)
    return model_data

def drops_sensitivity_final_layer(model_data, wgts, data_subset_frac):
    print('>>drops_sensitivity_final_layer()')
    scalar_vals = {}
    #would sometimes fail with error when scalar =0.1 so starting at 0.2
    for i in range(2, 11):
        vals = fit_final_layer(model_data, i/10, wgts, data_subset_frac)
        scalar_vals[i/10] = vals
    print(scalar_vals)
    return scalar_vals

def plot_multi_lines(x_list, y_list, n_runs, name):
    for x, y, in zip(x_list, y_list):
        plt.plot(x, y)
    plt.savefig(name)
    plt.close()


def plot_tuple_dict(d, run):
    lists = sorted(d.items())
    x, y = zip(*lists)
    y0 = [i[0] for i in y]
    y1 = [i[1] for i in y]

    plt.plot(x, y0)
    plt.savefig('drops_sens_final_layer_val_loss_{0}_{1}.png'.format(DATA_SUBSET, run))
    plt.close()

    plt.plot(x, y1)
    plt.savefig('drops_sens_final_layer_acc_{0}_{1}.png'.format(DATA_SUBSET, run))
    plt.close()



def fit_final_layer(model_data, drops_scalar, wgts, data_subset_frac, use_pt_wgts=True):
    print('>>fit_final_layer() drops_scalar: {0}'.format(drops_scalar))
    assert drops_scalar >= 0.1
    assert drops_scalar<=1.0
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*drops_scalar

    kwargs = {'dropouti': drops[0], 'dropout': drops[1], 'wdrop': drops[2], 'dropoute': drops[3], 'dropouth': drops[4]}

    #returns a RNN_Learner with model ~ SequentialRNN(RNN_Encoder(...), LinearDecoder(...))
    learner = model_data.get_model(opt_fn = opt_fn, emb_sz = EMBEDDING_SIZE, n_hid = n_hid, n_layers = n_layers, **kwargs)

    learner.metrics = [accuracy]
    learner.unfreeze

    if use_pt_wgts:
        learner.model.load_state_dict(wgts)

    lr=1e-3
    lrs=lr

    start = timer()
    #getting torch.backends.cudnn.CuDNNError: 8: b'CUDNN_STATUS_EXECUTION_FAILED' here
    #when going dropout parameter testing
    vals = learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)
    end = timer()
    print(f'elapsed: {end - start}')
    print(f'{vals}',flush=True)
    learner.save('lm_last_fit'+'_' + str(data_subset_frac))
    return vals, learner, lrs

def train_full_model_fast(learner, lrs, data_subset_frac):
    print('>>train_full_model()')
    learner.load('lm_last_fit'+'_' + str(data_subset_frac))

    learner.unfreeze()
    learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

    vals = learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=3)

    learner.save('lm1'+'_' + str(data_subset_frac))
    learner.save_encoder('lm1_enc'+'_' + str(data_subset_frac))
    return vals


def train_full_model(learner, lrs, data_subset_frac):
    print('>>train_full_model()')
    learner.load('lm_last_fit'+'_' + str(data_subset_frac))

    learner.unfreeze()

    start = timer()
    learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
    end = timer()
    print(f'elapsed: {end - start}')

    learner.sched.plot()
    plt.show()

    start = timer()
    learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)
    end = timer()
    print(f'elapsed: {end - start}')

    learner.save('lm1'+'_' + str(data_subset_frac))
    learner.save_encoder('lm1_enc'+'_' + str(data_subset_frac))

    learner.sched.plot_loss()
    plt.show()

def save_obj(obj, name ):
    with open('data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data_tests/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def classifier_tokens(data_subset_frac):
    # ## Classifier Tokens


    df_trn = pd.read_csv(str(CLAS_PATH)+'/train'+'_' + str(data_subset_frac)+'.csv', header=None, chunksize=CHUNKSIZE)
    df_val = pd.read_csv(str(CLAS_PATH)+'/test'+'_' + str(data_subset_frac)+'.csv', header=None, chunksize=CHUNKSIZE)

    tok_trn, trn_labels = get_all(tf_reader=df_trn, n_lbls=1)
    tok_val, val_labels = get_all(tf_reader=df_trn, n_lbls=1)

    (CLAS_PATH/'tmp').mkdir(exist_ok=True)

    np.save(str(CLAS_PATH)+'/tmp'/'tok_trn'+'_' + str(data_subset_frac)+'.npy', tok_trn)
    np.save(str(CLAS_PATH)+'/tmp'/'tok_val'+'_' + str(data_subset_frac)+'.npy', tok_val)

    np.save(str(CLAS_PATH)+'/tmp'/'trn_labels'+'_' + str(data_subset_frac)+'.npy', trn_labels)
    np.save(str(CLAS_PATH)+'/tmp'/'val_labels'+'_' + str(data_subset_frac)+'.npy', val_labels)

    tok_trn = np.load(str(CLAS_PATH)+'/tmp/tok_trn'+'_' + str(data_subset_frac)+'.npy')
    tok_val = np.load(str(CLAS_PATH)+'/tmp/tok_val'+'_' + str(data_subset_frac)+'.npy')

    itos = pickle.load(str(LM_PATH)+'/tmp/itos'+'_' + str(data_subset_frac)+'.pkl').open('rb')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)

    trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(str(CLAS_PATH)+'/tmp/trn_ids'+'_' + str(data_subset_frac)+'.npy', trn_clas)
    np.save(str(CLAS_PATH)+'/tmp/val_ids'+'_' + str(data_subset_frac)+'.npy', val_clas)


# ## Classifier

def test_dropout_stability():
    #run through this nx, look at variation over runs
    data_subset_list = [25, 10, 5, 1]
    for data_subset_frac in data_subset_list:
        trn_lm, val_lm, itos = get_lookups(data_subset_frac)
        wgts = wikitext103_conversion(itos, trn_lm, data_subset_frac)
        runs = 20
        trn_lm, val_lm, itos = get_lookups(data_subset_frac)
        vocab_size = len(itos)
        x_list = []
        y1_list = []
        y0_list= []
        for run in range(runs):
            #will be different each time
            model_data = language_model(trn_lm, val_lm, vocab_size)
            scalar_vals = drops_sensitivity_final_layer(model_data, wgts, data_subset_frac)
            lists = sorted(scalar_vals.items())
            x, y = zip(*lists)
            y0 = [i[0] for i in y]
            y1 = [i[1] for i in y]
            x_list.append(x)
            y0_list.append(y0)
            y1_list.append(y1)
        print(f'y0_list: {y0_list}, y1_list: {y1_list}')
        name_0 = 'drops_sens_final_layer_val_loss_{0}_{1}_runs.png'.format(DATA_SUBSET, runs)
        plot_multi_lines(x_list, y0_list, runs, name_0)
        name_1 = 'drops_sens_final_layer_acc_{0}_{1}_runs.png'.format(DATA_SUBSET, runs)
        plot_multi_lines(x_list, y1_list, runs, name_1)

def test_data_subset_sensitivity(drops_scalar=0.7, use_pt_wgts=True):
    data_sz_list = [200, 100, 50, 25, 10, 2, 1]
    val_dict = {}
    val_dict_full = {}
    for data_subset_frac in data_sz_list:
        start = timer()
        trn_texts, val_texts, col_names = create_class_files(data_subset_frac)
        create_train_test_files(trn_texts, val_texts, col_names, data_subset_frac)
        tok_trn, tok_val = create_tokens(data_subset_frac)
        create_token_lookups(tok_trn, tok_val, data_subset_frac)
        trn_lm, val_lm, itos = get_lookups(data_subset_frac)
        vocab_size = len(itos)
        wgts = wikitext103_conversion(itos, trn_lm, data_subset_frac)
        model_data = language_model(trn_lm, val_lm, vocab_size)
        vals, learner, lrs = fit_final_layer(model_data, drops_scalar, wgts, data_subset_frac, use_pt_wgts)
        val_dict[data_subset_frac] = vals
        #save as we go in case something happens
        if use_pt_wgts:
            save_obj(vals, 'data_subset_sens_last_{0}'.format(data_subset_frac))
        else:
            save_obj(vals, 'data_subset_sens_last_no_pt_{0}'.format(data_subset_frac))
        vals = train_full_model_fast(learner, lrs, data_subset_frac)
        val_dict_full[data_subset_frac] = vals
        if use_pt_wgts:
            save_obj(vals, 'data_subset_sens_full_3_epoch_{0}'.format(data_subset_frac))
        else:
            save_obj(vals, 'data_subset_sens_full_3_no_pt_epoch_{0}'.format(data_subset_frac))
        end = timer()
        print(f'run {data_subset_frac} complete')
        elapsed = end - start
        print(elapsed)

def test_bs_sensitivity(drops_scalar=0.7):
    bs_list = [64, 32, 16, 8, 1]
    data_subset_frac = 10
    val_dict = {}
    val_dict_full = {}
    for bs in bs_list:
        start = timer()
        trn_texts, val_texts, col_names = create_class_files(data_subset_frac)
        create_train_test_files(trn_texts, val_texts, col_names, data_subset_frac)
        tok_trn, tok_val = create_tokens(data_subset_frac)
        create_token_lookups(tok_trn, tok_val, data_subset_frac)
        trn_lm, val_lm, itos = get_lookups(data_subset_frac)
        vocab_size = len(itos)
        wgts = wikitext103_conversion(itos, trn_lm, data_subset_frac)
        model_data = language_model(trn_lm, val_lm, vocab_size, bs)
        vals, learner, lrs = fit_final_layer(model_data, drops_scalar, wgts, data_subset_frac)
        val_dict[bs] = vals
        #save as we go in case something happens
        save_obj(vals, 'data_subset_sens_last_{0}_bs_{1}'.format(data_subset_frac, bs))
        vals = train_full_model_fast(learner, lrs, data_subset_frac)
        val_dict_full[bs] = vals
        save_obj(vals, 'data_subset_sens_full_3_epoch_{0}_bs_{1}'.format(data_subset_frac, bs))
        end = timer()
        print(f'run {data_subset_frac} complete')
        elapsed = end - start
        print(elapsed)

    save_obj(val_dict, 'data_subset_sens_last_bs')
    save_obj(val_dict, 'data_subset_sens_full_3_epoch_bs')


def workflow():
    start = timer()
    #only need to run these once
    #trn_texts, val_texts, col_names = create_class_files()
    #create_train_test_files(trn_texts, val_texts, col_names)
    #tok_trn, tok_val = create_tokens()
    #create_token_lookups(tok_trn, tok_val)
    #trn_lm, val_lm, itos = get_lookups(data_subset_frac=50)
    #vocab_size = len(itos)
    #wgts = wikitext103_conversion(itos, trn_lm, data_subset_frac=50)
    #model_data = language_model(trn_lm, val_lm, vocab_size)
    #fit_final_layer(model_data)

    # data subset workflow
    test_data_subset_sensitivity(drops_scalar=0.7, use_pt_wgts=False)

    #dropout_stability workflow
    test_dropout_stability()

    # batch size workflow
    test_bs_sensitivity()

    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()