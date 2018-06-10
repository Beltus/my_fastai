
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

#PATH=Path('../..')/'data/lm/models/wikitext-103/'
PATH=Path('../..')/'data/lm/models/wikitext-2/'
#PATH=Path('../..')/'data/lm/models/penn/'
PATH_TMP = PATH/'tmp'

#LM_PATH=Path('../..')/'data/lm/models/wikitext-103/'
LM_PATH=Path('../..')/'data/lm/models/wikitext-2/'
#LM_PATH=Path('../..')/'data/lm/models/penn/'
LM_PATH.mkdir(exist_ok=True)
LM_PATH_TMP = LM_PATH/'tmp'

chunksize=24000

max_vocab = 60000

min_freq = 2

em_sz,nh,nl = 400,1150,3

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

re1 = re.compile(r'  +')
#lang_mdl = 'penn'
lang_mdl = 'wikitext2'
#lang_mdl = 'wikitext-103'

def save_obj(obj, name ):
    with open('../data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_texts(df):
    texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok




def create_train_test_files(trn_texts, val_texts, col_names):
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




def create_tokens():
    df_trn = pd.read_csv(PATH + 'train.txt', header=None)
    df_val = pd.read_csv(PATH + 'valid.txt', header=None)

    tok_trn = get_texts(df_trn)
    tok_val = get_texts(df_val)
    np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
    np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)



def get_tokens():
    tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
    tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')
    return tok_trn, tok_val


def create_token_lookups(tok_trn, tok_val):

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
    trn_lm = np.load(PATH/'tmp'/'trn_ids.npy')
    val_lm = np.load(PATH/'tmp'/'val_ids.npy')
    itos = pickle.load(open(PATH/'tmp'/'itos.pkl', 'rb'))
    return trn_lm, val_lm, itos

def pretrained_conversion(itos, trn_lm, pt_model = 'wt103', save_wgts=True):
    vs=len(itos)
    vs,len(trn_lm)

    PRE_PATH = PATH/'models'/pt_model
    PRE_LM_PATH = PRE_PATH/'fwd_{0}.h5'.format(pt_model)

    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)

    itos2 = pickle.load((PRE_PATH/'itos_{0}.pkl'.format(pt_model)).open('rb'))
    stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

    new_w = np.zeros((vs, em_sz), dtype=np.float32)
    for i,w in enumerate(itos):
        r = stoi2[w]
        new_w[i] = enc_wgts[r] if r>=0 else row_m

    wgts['0.encoder.weight'] = T(new_w)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    wgts['1.decoder.weight'] = T(np.copy(new_w))
    if save_wgts:
        pickle.dump(wgts, open(str(PRE_PATH)+'/wgts_{0}.pkl'.format(pt_model),'wb'))
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
    save_obj(ep_vals, f'{lang_mdl}_ep_vals_final_{run_id}')
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
    save_obj(ep_vals, f'{lang_mdl}_ep_vals_full_{run_id}')

def create_learner(md, run_id='', dropouti=0.25, dropout=0.1, wdrop=0.2,
                    dropoute=0.02, dropouth=0.15, use_pt_wgts=True):
    #create a learner without fitting final layer
    print('>>create_learner')

    learner= md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth)

    learner.metrics = [accuracy]

    lrs = 1e-3
    return learner, lrs

def train_full_model_vanilla(learner, lrs, lang_mdl, run_id='', cycle_len=12, use_clr_beta=True):
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
    save_obj(elapsed, f'{lang_mdl}_ep_vals_use_clr_beta_{use_clr_beta}_{run_id}')

    # We save the trained model weights and separately save the encoder part of the LM model as well. This will serve as our backbone in the classification task model.
    learner.save('lm1'+'_' + str(run_id))
    learner.save_encoder('lm1_enc'+'_' + str(run_id))
    save_obj(ep_vals, f'{lang_mdl}_ep_vals_full_{run_id}')

def test_dropout_parameters():
    #all other dropout params set to 0
    #run through this nx, look at variation over runs
    trn_lm, val_lm, itos = get_lookups()
    #wgts = pretrained_conversion(itos, trn_lm, save_wgts=False)

    vocab_size = len(itos)

    md = language_model(trn_lm, val_lm, vocab_size)

    drops_scalar = 1.0
    range_strt = 1
    range_stop = 10
    prefix = 'all_drop_0'
    for drop_base in [0.001, 0.01, 0.02, 0.05, 0.1, 0.5]:
        prefix = f'di_d_de_dh_drop_0_dw_{str(drop_base)}'
        #d is OK as all other params 0
        #for drop in ['d', 'wd', 'de', 'dh', 'di']:
        for drop in ['wd', 'de', 'dh', 'di']:
            dropouti = 0
            dropout = 0
            dropoute = 0
            dropouth = 0
            # Cant set this to zero
            wdrop = drop_base
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
                learner, lrs = create_learner(md, run_id=run_id, dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth, use_pt_wgts=True)
                train_full_model(learner, lrs, run_id, cycle_len=12, use_clr_beta=False)

def best_dropout_parameters():
    #all other dropouts set to best value found using test_dropout_parameters()
    #run through this nx, look at variation over runs
    trn_lm, val_lm, itos = get_lookups()
    #wgts = pretrained_conversion(itos, trn_lm, save_wgts=False)

    vocab_size = len(itos)

    md = language_model(trn_lm, val_lm, vocab_size)

    drops_scalar = 1.0
    range_strt = 6
    range_stop = 10
    for drop_base in [0.001, 0.01, 0.02, 0.05, 0.1, 0.5]:
        #NB in some runs ran with di, d, de, dh set to 0 below instead of using test_dropout_parameters()
        prefix = f'di_0.2_d_0.7_de_0.1_dh_0.3_dw_{str(drop_base)}'
        #for drop in ['d', 'wd', 'de', 'dh', 'di']:
        for drop in ['wd']:
            dropouti = 0.2
            dropout = 0.7
            dropoute = 0.1
            dropouth = 0.3
            # Cant set this to zero
            wdrop = drop_base
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
                learner, lrs = create_learner(md, run_id=run_id, dropouti=dropouti, dropout=dropout, wdrop=wdrop, dropoute=dropoute, dropouth=dropouth, use_pt_wgts=True)
                train_full_model(learner, lrs, run_id, cycle_len=12, use_clr_beta=False)


def clr_testing():
    trn_lm, val_lm, itos = get_lookups()
    #wgts = pretrained_conversion(itos, trn_lm, save_wgts=False)

    vocab_size = len(itos)
    drops_scalar = 1.0
    range_strt = 1
    range_stop = 10
    prefix = 'all_drop_0'
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
    learner, lrs = create_learner(md, run_id=run_id, dropouti=dropouti, dropout=dropout, wdrop=wdrop,
                                  dropoute=dropoute, dropouth=dropouth, use_pt_wgts=True)
    train_full_model_vanilla(learner, lrs, lang_mdl, run_id, cycle_len=1, use_clr_beta=True)
    train_full_model_vanilla(learner, lrs, lang_mdl, run_id, cycle_len=1, use_clr_beta=False)

def workflow():
    start = timer()
    # only need to run these once
    #tok_trn, tok_val = create_tokens()
    #create_token_lookups(tok_trn, tok_val)
    #trn_lm, val_lm, itos = get_lookups()
    #vocab_size = len(itos)
    #clr_testing()
    best_dropout_parameters()
    #test_dropout_parameters()
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()

