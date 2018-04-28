
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

#PATH=Path('..')/'data/imdb/aclImdb'
#os.listdir(PATH)


#Language Model
LM_PATH=Path('../..')/'data/awdlstm/data/wikitext-2/'
LM_PATH.mkdir(exist_ok=True)

#unsup for unlabelled
CLASSES = ['neg', 'pos', 'unsup']

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

def create_tokens():
    # ## Language Model Tokens
    #
    # Turn text into a a list of tokens using Spacy

    print(f'max thread count: {len(os.sched_getaffinity(0))}')

    df_trn = pd.read_csv(str(LM_PATH)+'/train.txt', sep='delimiter', header=None)
    df_val = pd.read_csv(str(LM_PATH)+'/test.txt', sep='delimiter', header=None)

    print(f'df_trn: {df_trn.head()}')
    print(f'df_val: {df_val.head()}')

    tok_trn = df_trn.ix[:, 0].tolist()
    tok_val = df_val.ix[:, 0].tolist()

    (LM_PATH/'tmp').mkdir(exist_ok=True)

    np.save(str(LM_PATH)+'/tmp/tok_trn.npy', tok_trn)
    np.save(str(LM_PATH)+'/tmp/tok_val.npy', tok_val)
    return tok_trn, tok_val


def create_token_lookups(tok_trn, tok_val):
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

    np.save(str(LM_PATH)+'/tmp/trn_ids.npy', trn_lm)
    np.save(str(LM_PATH)+'/tmp/val_ids.npy', val_lm)
    pickle.dump(itos, open(str(LM_PATH)+'/tmp/itos.pkl','wb'))

def get_lookups():
    print('>>get_lookups()')
    trn_lm = np.load(str(LM_PATH)+'/tmp/trn_ids.npy')
    val_lm = np.load(str(LM_PATH)+'/tmp/val_ids.npy')
    itos = pickle.load(open(str(LM_PATH)+'/tmp/itos.pkl', 'rb'))
    return trn_lm, val_lm, itos

def language_model(trn_lm, val_lm, vocab_size, bs=52):
    print('>>language_model()')
    # ## Language Model

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
    #path, pad_idx, n_tok, trn_dl, val_dl, test_dl=None, bptt=70, backwards=False, **kwargs
    model_data = LanguageModelData(LM_PATH, pad_idx=1, n_tok=vocab_size, trn_dl=trn_dl, val_dl=val_dl, bs=bs, bptt=bptt)
    return model_data

def drops_sensitivity_final_layer(model_data, wgts):
    print('>>drops_sensitivity_final_layer()')
    scalar_vals = {}
    #would sometimes fail with error when scalar =0.1 so starting at 0.2
    for i in range(2, 11):
        vals = fit_final_layer(model_data, i/10, wgts)
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
    plt.savefig('drops_sens_final_layer_val_loss_{0}.png'.format(run))
    plt.close()

    plt.plot(x, y1)
    plt.savefig('drops_sens_final_layer_acc_{0}.png'.format(run))
    plt.close()

def fit_final_layer(model_data):
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

    kwargs = {'dropouti': drops[0], 'dropout': drops[1], 'wdrop': drops[2], 'dropoute': drops[3], 'dropouth': drops[4]}

    #returns a RNN_Learner with model ~ SequentialRNN(RNN_Encoder(...), LinearDecoder(...))
    learner = model_data.get_model(opt_fn = opt_fn, emb_sz = EMBEDDING_SIZE, n_hid = n_hid, n_layers = n_layers, **kwargs)

    learner.metrics = [accuracy]
    learner.unfreeze

    lr=1e-3
    lrs=lr

    start = timer()
    #getting torch.backends.cudnn.CuDNNError: 8: b'CUDNN_STATUS_EXECUTION_FAILED' here
    #when going dropout parameter testing
    vals = learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)
    end = timer()
    print(f'elapsed: {end - start}')
    print(f'{vals}',flush=True)
    learner.save('lm_wt2_last_fit')
    return vals, learner, lrs

def train_full_model_fast(learner, lrs):
    print('>>train_full_model()')
    learner.load('lm_wt2_last_fit')

    learner.unfreeze()
    learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

    vals = learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=3)

    learner.save('lm_wt2')
    learner.save_encoder('lm_wt2_enc')
    return vals


def train_full_model(learner, lrs):
    print('>>train_full_model()')
    learner.load('lm_wt2_last_fit')

    learner.unfreeze()

    start = timer()
    learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
    end = timer()
    print(f'elapsed: {end - start}')

    #learner.sched.plot()
    #plt.show()

    start = timer()
    learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)
    end = timer()
    print(f'elapsed: {end - start}')

    learner.save('lm_wt2')
    learner.save_encoder('lm_wt2_enc')

    learner.sched.plot_loss()
    plt.show()


def workflow():
    start = timer()

    tok_trn, tok_val = create_tokens()
    create_token_lookups(tok_trn, tok_val)
    trn_lm, val_lm, itos = get_lookups()
    vocab_size = len(itos)
    #wgts = wikitext2_conversion(itos, trn_lm)
    model_data = language_model(trn_lm, val_lm, vocab_size)
    vals, learner, lrs = fit_final_layer(model_data)
    #vals = train_full_model_fast(learner, lrs)
    train_full_model(learner, lrs)

    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()