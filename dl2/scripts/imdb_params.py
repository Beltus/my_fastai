
# coding: utf-8

#dropout parameter testing

from timeit import default_timer as timer
import html
import torch
import sys

import matplotlib.pylab as plt

from fastai.text import *

#1080ti
torch.cuda.set_device(0)
#1080 on xeon
#torch.cuda.set_device(1)
torch.cuda.current_device()

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH=Path('../..')/'data/imdb/aclImdb/'


# ## Standardize format

# In[5]:


CLAS_PATH=Path('../..')/'data/imdb/imdb_clas/'
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('../..')/'data/imdb/imdb_lm/'
LM_PATH.mkdir(exist_ok=True)

CLASSES = ['neg', 'pos', 'unsup']

chunksize=24000

max_vocab = 60000

min_freq = 2

em_sz,nh,nl = 400,1150,3

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))



re1 = re.compile(r'  +')


def save_obj(obj, name ):
    with open('../data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

def create_class_files(data_subset_frac):

    trn_texts,trn_labels = get_texts(PATH/'train')
    val_texts,val_labels = get_texts(PATH/'test')

    len(trn_texts),len(val_texts)

    col_names = ['labels','text']

    np.random.seed(42)
    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))

    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]

    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]

    df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

    df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)
    df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)

    (CLAS_PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)
    return trn_texts, val_texts, col_names


def create_train_test_files(trn_texts, val_texts, col_names, data_subset_frac):
    trn_texts,val_texts = sklearn.model_selection.train_test_split(
        np.concatenate([trn_texts,val_texts]), test_size=0.1)

    len(trn_texts), len(val_texts)

    df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

    df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
    df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def lookup_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


def create_tokens(data_subset_frac):

    df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)

    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)

    (LM_PATH/'tmp').mkdir(exist_ok=True)

    np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
    np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)



def get_tokens():
    tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
    tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')
    return tok_trn, tok_val


def create_token_lookups(tok_trn, tok_val, data_subset_frac):

    freq = Counter(p for o in tok_trn for p in o)
    freq.most_common(25)

    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')

    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)

    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
    np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
    pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))

def get_lookups():
    trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
    val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
    itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))
    return trn_lm, val_lm, itos

def wikitext103_conversion(itos, trn_lm, save_wgts=True):
    vs=len(itos)
    vs,len(trn_lm)

    PRE_PATH = PATH/'models'/'wt103'
    PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)

    itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
    stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

    new_w = np.zeros((vs, em_sz), dtype=np.float32)
    for i,w in enumerate(itos):
        r = stoi2[w]
        new_w[i] = enc_wgts[r] if r>=0 else row_m

    wgts['0.encoder.weight'] = T(new_w)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    wgts['1.decoder.weight'] = T(np.copy(new_w))
    if save_wgts:
        pickle.dump(wgts, open(str(PRE_PATH)+'/wgts_wt103.pkl','wb'))
    print(f'wgts.keys: {wgts.keys()}')
    return wgts

def language_model(trn_lm, val_lm, vocab_size, bs=52):

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
    md = LanguageModelData(PATH, 1, vocab_size, trn_dl, val_dl, bs=bs, bptt=bptt)
    return md



def fit_final_layer(md, wgts, run_id='', dropouti=0.25, dropout=0.1, wdrop=0.2,
                    dropoute=0.02, dropouth=0.15, use_pt_wgts=True):

    # We setup the dropouts for the model - these values have been chosen after experimentation. If you need to update them for custom LMs, you can change the weighting factor (0.7 here) based on the     amount of data you have. For more data, you can reduce dropout factor and for small datasets, you can reduce overfitting by choosing a higher dropout factor. *No other dropout value requires tuning*

    #drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


    # We first tune the last embedding layer so that the missing tokens initialized with mean weights get tuned properly. So we freeze everything except the last layer.
    #
    # We also keep track of the *accuracy* metric.
    print('>>fit_final_layer')

    learner= md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth)

    learner.metrics = [accuracy]
    destination = learner.model.state_dict()
    learner.freeze_to(-1)
    #print(f'model state: {destination.keys()}')
    #print(f'wgts keys: {wgts.keys()} \n')

    if use_pt_wgts:
        destination = learner.model.state_dict()
        #print(f'model keys: {destination.keys()}')
        #print(f'wgts keys: {wgts.keys()}')
        learner.model.load_state_dict(wgts)
        #print('wgts loaded')

    lr=1e-3
    lrs = lr

    vals, ep_vals = learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1, get_ep_vals=True)
    learner.save('lm_last_fit'+'_'+str(run_id))
    save_obj(ep_vals, 'imdb_ep_vals_final_' + run_id)
    return learner, lrs


def train_full_model(learner, lrs, run_id='', cycle_len=12, use_clr_beta=True, use_last_pt=False):
    print('>>train_full_model')

    if use_last_pt:
        learner.load('lm_last_fit'+'_'+str(run_id))
    learner.unfreeze()

    start = timer()
    if use_clr_beta:
        vals, ep_vals = learner.fit(lrs, 1, wds=wd, use_clr_beta=(10,10,0.95,0.85), cycle_len=cycle_len, get_ep_vals=True)
    else:
        vals, ep_vals = learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=cycle_len, get_ep_vals=True)
    end = timer()
    elapsed = end - start
    print(f'--train_full_model() fit() took {elapsed}sec')

    # We save the trained model weights and separately save the encoder part of the LM model as well. This will serve as our backbone in the classification task model.
    learner.save('lm1'+'_' + str(run_id))
    learner.save_encoder('lm1_enc'+'_' + str(run_id))
    save_obj(ep_vals, 'imdb_ep_vals_full_' + str(run_id))

def create_learner(md, wgts, run_id='', dropouti=0.25, dropout=0.1, wdrop=0.2,
                    dropoute=0.02, dropouth=0.15, use_pt_wgts=True):
    #create a learner without fitting final layer
    print('>>create_learner')

    learner= md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth)

    learner.metrics = [accuracy]
    if use_pt_wgts:
        learner.model.load_state_dict(wgts)

    lrs = 1e-3
    return learner, lrs

def train_full_model_vanilla(learner, lrs, run_id='', cycle_len=12, use_clr_beta=True):
    print('>>train_full_model_vanilla')

    learner.unfreeze()

    start = timer()
    if use_clr_beta:
        vals, ep_vals = learner.fit(lrs, 1, wds=wd, use_clr_beta=(10,10,0.95,0.85), cycle_len=cycle_len, get_ep_vals=True)
    else:
        vals, ep_vals = learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=cycle_len, get_ep_vals=True)
    end = timer()
    elapsed = end - start
    print(f'--train_full_model() fit() took {elapsed}sec')
    save_obj(elapsed, f'imdb_ep_vals_use_clr_beta_{use_clr_beta}_{run_id}')

    # We save the trained model weights and separately save the encoder part of the LM model as well. This will serve as our backbone in the classification task model.
    learner.save('lm1'+'_' + str(run_id))
    learner.save_encoder('lm1_enc'+'_' + str(run_id))
    save_obj(ep_vals, 'imdb_ep_vals_full_' + str(run_id))

def classifier_tokens():
    # ## Classifier tokens

    # The classifier model is basically a linear layer custom head on top of the LM backbone. Setting up the classifier data is similar to the LM data setup except that we cannot use the unsup movie reviews this time.

    df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)

    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)

    (CLAS_PATH/'tmp').mkdir(exist_ok=True)

    np.save(CLAS_PATH/'tmp'/'tok_trn.npy', tok_trn)
    np.save(CLAS_PATH/'tmp'/'tok_val.npy', tok_val)

    np.save(CLAS_PATH/'tmp'/'trn_labels.npy', trn_labels)
    np.save(CLAS_PATH/'tmp'/'val_labels.npy', val_labels)

    tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
    tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')

    itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)

    trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
    np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)

def classifier(itos):
    # ## Classifier
    trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
    val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')

    trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
    val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))

    bptt,em_sz,nh,nl = 70,400,1150,3
    vs = len(itos)
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 48

    min_lbl = trn_labels.min()
    trn_labels -= min_lbl
    val_labels -= min_lbl
    c=int(trn_labels.max())+1

    trn_ds = TextDataset(trn_clas, trn_labels)
    val_ds = TextDataset(val_clas, val_labels)
    trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
    val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)

    # part 1
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5

    m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25.
    learn.metrics = [accuracy]

    lr=3e-3
    lrm = 2.6
    lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])

    lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

    wd = 1e-7
    wd = 0
    learn.load_encoder('lm2_enc')

    learn.freeze_to(-1)
    learn.lr_find(lrs/1000)
    learn.sched.plot()
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
    learn.save('clas_0')
    learn.load('clas_0')
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
    learn.save('clas_1')
    learn.load('clas_1')
    learn.unfreeze()
    learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))
    learn.sched.plot_loss()

    learn.save('clas_2')
    # The previous state of the art result was 94.1% accuracy (5.9% error). With bidir we get 95.4% accuracy (4.6% error).
    learn.sched.plot_loss()

def test_dropout_parameters():
    #run through this nx, look at variation over runs
    trn_lm, val_lm, itos = get_lookups()
    wgts = wikitext103_conversion(itos, trn_lm, save_wgts=False)

    vocab_size = len(itos)

    md = language_model(trn_lm, val_lm, vocab_size)

    drops_scalar = 1.0
    range_strt = 1
    range_stop = 10
    prefix = 'all_drop_0'
    dropouti= 0
    dropout=0
    dropoute=0
    dropouth=0
    #Cant set this to zero
    #wdrop=0
    for drop_base in [0.001, 0.01, 0.02, 0.05, 0.1, 0.5]:
        wdrop = drop_base
        prefix = f'di_d_de_dh_drop_0_dw_{str(drop_base)}'
        for drop in ['d', 'wd', 'de', 'dh', 'di']:
            dropouti=0
            dropout=0
            dropoute=0
            dropouth=0
            for i in range(range_strt, range_stop):
                drop_val = i / 10
                run_id = f'{prefix}_{drop}_{str(drop_val)}'
                print(f'{run_id}')
                if drop == 'di':
                    dropouti = drop_val*drops_scalar
                elif drop == 'd':
                    dropout = drop_val*drops_scalar
                elif drop == 'wd':
                    wdrop = drop_val*drops_scalar
                elif drop == 'de':
                    dropoute = drop_val*drops_scalar
                elif drop == 'dh':
                    dropouth = drop_val*drops_scalar
                #model_data, drops_scalar, wgts, data_subset_frac, run_id='', dropouti=0.25, dropout=0.1, wdrop=0.2, dropoute=0.02, dropouth=0.15, use_pt_wgts=True
                learner, lrs = create_learner(md, wgts, run_id=run_id, dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth, use_pt_wgts=True)
                train_full_model(learner, lrs, run_id, cycle_len=12, use_clr_beta=False)

def clr_testing():
    trn_lm, val_lm, itos = get_lookups()
    wgts = wikitext103_conversion(itos, trn_lm, save_wgts=False)

    vocab_size = len(itos)
    drops_scalar = 1.0
    range_strt = 1
    range_stop = 10
    prefix = 'clr_all_drop_0'
    dropouti=0
    dropout=0
    dropoute=0
    dropouth=0

    dropout = 0.1
    drop = 'di'
    drop_base = 0.001
    wdrop = drop_base
    run_id = f'{prefix}_{drop}_{str(0.1)}'
    prefix = f'di_d_de_dh_drop_0_dw_{str(drop_base)}'
    md = language_model(trn_lm, val_lm, vocab_size)
    learner, lrs = create_learner(md, wgts, run_id=run_id, dropouti=dropouti, dropout=dropout, wdrop=wdrop,
                                  dropoute=dropoute, dropouth=dropouth, use_pt_wgts=True)
    train_full_model_vanilla(learner, lrs, run_id, cycle_len=1, use_clr_beta=True)
    train_full_model_vanilla(learner, lrs, run_id, cycle_len=1, use_clr_beta=False)

def workflow():
    start = timer()
    clr_testing()
    test_dropout_parameters()
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()

