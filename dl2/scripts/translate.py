# coding: utf-8

from timeit import default_timer as timer
from fastai.text import *
from pathlib import Path
torch.cuda.set_device(0)
torch.cuda.current_device()


# ## Translate French to English

# ## Translation files
#
# Lesson 11 start 22:48
#
#
#

# French/English parallel texts from http://www.statmt.org/wmt15/translation-task.html


# google tranlate model has 8 layers, here we use 2
#
# we are using a cut down version - translate French questions. So look for things starting with Wh and ensing with ?

PATH = Path('../..')/'data/translate'
TMP_PATH = PATH/'tmp'
TMP_PATH.mkdir(exist_ok=True)
GLOVE_PATH = PATH/'glove.6B'
fname='giga-fren.release2.fixed'
en_fname = PATH/f'{fname}.en'
fr_fname = PATH/f'{fname}.fr'

re_eq = re.compile('^(Wh[^?.!]+\?)')
re_fq = re.compile('^([^?.!]+\?)')

#runs in notebook with bs=370, here reducing as encountered oom errors
bs=200

#why are we using 256 for number of hidden?
nh = 256
nl = 2

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

def create_fr_en_qs():
    lines = ((re_eq.search(eq), re_fq.search(fq))
         for eq, fq in zip(open(en_fname, encoding='utf-8'), open(fr_fname, encoding='utf-8')))

    qs = [(e.group(), f.group()) for e,f in lines if e and f]

    pickle.dump(qs, (PATH/'fr-en-qs.pkl').open('wb'))

def load_fr_en_qs():
    qs = pickle.load((PATH/'fr-en-qs.pkl').open('rb'))
    # Example sentence pairs
    print(f'{qs[:5]}, {len(qs)}')
    en_qs,fr_qs = zip(*qs)
    print(f'{fr_qs[0]}')
    return en_qs, fr_qs

def get_en_fr_tokens(en_qs, fr_qs):
    en_tok = Tokenizer.proc_all_mp(partition_by_cores(en_qs))

    #fr_tok = Tokenizer.proc_all_mp(partition_by_cores(fr_qs), 'fr')
    #to use less RAM use Tokenizer.proc_all or use less cores
    fr_tok = Tokenizer.proc_all(fr_qs, 'fr')
    print(f'{len(en_tok)}, {len(fr_tok)}')
    return en_tok, fr_tok

def remove_tail_end(en_tok, fr_tok):
    #Compute the 90th percentile of the word lengths
    np.percentile([len(o) for o in en_tok], 90), np.percentile([len(o) for o in fr_tok], 90)

    #remove the very largest sentences
    keep = np.array([len(o)<30 for o in en_tok])

    en_tok = np.array(en_tok)[keep]
    fr_tok = np.array(fr_tok)[keep]
    return en_tok, fr_tok

def save_tokens(en_tok, fr_tok):
    pickle.dump(en_tok, (PATH/'en_tok.pkl').open('wb'))
    pickle.dump(fr_tok, (PATH/'fr_tok.pkl').open('wb'))

def load_tokens():
    en_tok = pickle.load((PATH/'en_tok.pkl').open('rb'))
    fr_tok = pickle.load((PATH/'fr_tok.pkl').open('rb'))
    print(f'{en_tok[0]}, {fr_tok[0]}')
    return en_tok, fr_tok


def toks2ids(tok,pre):
    freq = Counter(p for o in tok for p in o)
    #get list of every word (limit at 40k)
    itos = [o for o,c in freq.most_common(40000)]
    itos.insert(0, '_bos_')
    itos.insert(1, '_pad_')
    itos.insert(2, '_eos_')
    itos.insert(3, '_unk')
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    ids = np.array([([stoi[o] for o in p] + [2]) for p in tok])
    np.save(TMP_PATH/f'{pre}_ids.npy', ids)
    pickle.dump(itos, open(TMP_PATH/f'{pre}_itos.pkl', 'wb'))
    return ids,itos,stoi

def convert_ids(en_tok, fr_tok):
    en_ids,en_itos,en_stoi = toks2ids(en_tok,'en')
    fr_ids,fr_itos,fr_stoi = toks2ids(fr_tok,'fr')
    return en_ids,en_itos,en_stoi, fr_ids,fr_itos,fr_stoi

def load_ids(pre):
    ids = np.load(TMP_PATH/f'{pre}_ids.npy')
    itos = pickle.load(open(TMP_PATH/f'{pre}_itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    return ids,itos,stoi

def get_ids():
    en_ids,en_itos,en_stoi = load_ids('en')
    fr_ids,fr_itos,fr_stoi = load_ids('fr')
    # [fr_itos[o] for o in fr_ids[0]], len(en_itos), len(fr_itos)
    return en_ids,en_itos,en_stoi, fr_ids,fr_itos,fr_stoi

# ## Word vectors
# 
# Rather than use seq to seq LM, we are going to use word vectors

def pickle_glove_data():

    with (GLOVE_PATH/'glove.6B.100d.txt').open('r', encoding='utf-8') as f: lines = [line.split() for line in f]
    en_vecd = {w:np.array(v, dtype=np.float32) for w,*v in lines}
    pickle.dump(en_vecd, open(GLOVE_PATH/'glove.6B.100d.dict.pkl','wb'))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError: return False

#deprecated - uses >64GB RAM
def get_vecs(lang):
    print('>>get_vecs() {0}'.format(lang))
    with (PATH/f'fastText/wiki.{lang}.vec').open('r', encoding='utf-8') as f:
        lines = [line.split() for line in f]
    lines.pop(0)
    vecd = {w:np.array(v, dtype=np.float32)
            for w,*v in lines if is_number(v[0]) and len(v)==300}
    pickle.dump(vecd, open(PATH/f'fastText/wiki.{lang}.pkl','wb'))
    print('<<get_vecs()')
    return vecd

def get_vecsb(lang):
    vecd = {}
    with open(PATH/f'fastText/wiki.{lang}.vec', encoding='utf-8') as infile:
        length, dim = infile.readline().split()
        for i in tqdm(range(int(length))):
            line = infile.readline()
            while len(line) == 0:
                line = infile.readline()
            w, *v = line.split()
            if is_number(v[0]) and len(v)==300:
                vecd[w] = np.array(v, dtype=np.float32)
    pickle.dump(vecd, open(PATH/f'fastText/wiki.{lang}.pkl','wb'))
    return vecd

def write_vectors():
    #this takes a few minutes
    en_vecd = get_vecsb('en')
    fr_vecd = get_vecsb('fr')

def load_vectors():
    en_vecd = pickle.load(open(PATH/'fastText/wiki.en.pkl','rb'))
    fr_vecd = pickle.load(open(PATH/'fastText/wiki.fr.pkl','rb'))
    #300 dimensional en and fr word embeddings
    dim_en_vec = len(en_vecd[','])
    dim_fr_vec = len(fr_vecd[','])
    return en_vecd, fr_vecd, dim_en_vec, dim_fr_vec



def get_stacked_en_vectors():
    en_vecs = np.stack(list(en_vecd.values()))
    #mean is near 0, std_dev ~0.3
    print(f'{en_vecs.mean()},{en_vecs.std()}')
    return en_vecs


# ## Model Data

# Often corpuses have a long lail of sequences - these can overwhelm how long things take.
# We want to truncate those longer words
#

def clip_en_fr_ids(en_ids, fr_ids, en_clip=99, fr_clip=98):

    enlen_99 = int(np.percentile([len(o) for o in en_ids], 99))
    frlen_98 = int(np.percentile([len(o) for o in fr_ids], 98))

    en_ids_tr = np.array([o[:enlen_99] for o in en_ids])
    fr_ids_tr = np.array([o[:frlen_98] for o in fr_ids])
    return en_ids_tr, fr_ids_tr, enlen_99, frlen_98


#pytorch Dataset requires 2 things - a length and a indexer
class GeneralDataset(Dataset):
    def __init__(self, x, y): 
        self.x,self.y = x,y
        
    def __getitem__(self, idx): 
        #return a numpy array of tuples
        return A(self.x[idx], self.y[idx])
    
    def __len__(self): 
        return len(self.x)

def create_train_val_sets(en_ids_tr, fr_ids_tr):
    #create training and validation set. index into bool arrays
    np.random.seed(42)
    trn_keep = np.random.rand(len(en_ids_tr))>0.1
    en_trn,fr_trn = en_ids_tr[trn_keep],fr_ids_tr[trn_keep]
    en_val,fr_val = en_ids_tr[~trn_keep],fr_ids_tr[~trn_keep]
    return en_trn,fr_trn, en_val,fr_val

def create_datasets(fr_trn,en_trn, fr_val,en_val):
    #switch these two around tp translate en->fr
    trn_ds = GeneralDataset(fr_trn,en_trn)
    val_ds = GeneralDataset(fr_val,en_val)
    return trn_ds, val_ds

def sortish_sample_en(en_trn, en_val):
    #make more time & memory efficient, sortish sampled on length
    trn_samp = SortishSampler(en_trn, key=lambda x: len(en_trn[x]), bs=bs)
    val_samp = SortSampler(en_val, key=lambda x: len(en_val[x]))
    #trn_ds.x[0]
    return trn_samp, val_samp


def create_dataloaders(trn_ds, val_ds, trn_samp, val_samp):
    trn_dl = DataLoader(trn_ds, bs, transpose=True, transpose_y=True, num_workers=1, pad_idx=1, pre_pad=False, sampler=trn_samp)
    val_dl = DataLoader(val_ds, int(bs*1.6), transpose=True, transpose_y=True, num_workers=1, pad_idx=1, pre_pad=False, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)
    return trn_dl, val_dl, md

#it = iter(trn_dl)
#its = [next(it) for i in range(5)]
#[(len(x),len(y)) for x,y in its]

def save_obj(obj, name ):
    with open('../data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data_tests/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ## Initial model
# 
# Start with the Encoder

def create_emb(vecs, itos, em_sz):
    #vecs: pretrained vectors
    #itos: note for Embedding rows must equal vocab size
    #em_sz: determined by fastText (300) 
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    #learnable objects have a weight attrib which has a data attrib (type tensor)
    wgts = emb.weight.data
    len_wgts = len(wgts)
    miss = []
    #so can start out using fastText embeddings we replace the random embedddings above with pretrained ones
    for i,w in enumerate(itos):
        try: 
            #our pre-trained weights have a std dev of ~0.3, random wghts have std dev of ~1
            wgts[i] = torch.from_numpy(vecs[w]*3)
        except: 
            #keep track of ones we dont find
            miss.append(w)
    print(f'len(wgts): {len_wgts}, len(missed): {len(miss)}, miss[5:10]: {miss[5:10]}, % missed: {(len(miss)/len_wgts)*100}')
    return emb


class Seq2SeqRNNAWD(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2,
                 rnn_type='GRU', rnn_enc_drop=0.25, rnn_dec_drop=0.1, emb_enc_drop=0.15, out_drop=0.35):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl, self.nh, self.out_sl = nl, nh, out_sl
        #ntoken, emb_sz, nhid, nlayers, pad_token, bidir=False,
        #dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5
        #TODO check ntoken, pad_token are correct
        self.rnn_enc = RNN_Encoder(ntoken=vecs_enc, em_sz_enc=em_sz_enc, nh=nh, num_layers=nl, pad_token=1, dropout=rnn_enc_drop)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.rnn_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=rnn_dec_drop)
        self.emb_enc_drop = nn.Dropout(emb_enc_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

        #Merity et al. variable naming
        self.dropout = out_drop
        self.dropouti = rnn_dec_drop
        self.dropouth = rnn_enc_drop
        self.dropoute = emb_enc_drop

    def forward(self, inp):
        sl, bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.rnn_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.rnn_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp == 1).all():
                break
        return torch.stack(res)

    def initHidden(self, bs):
        zero_tensor = V(torch.zeros(self.nl, bs, self.nh))
        return zero_tensor

class Seq2SeqRNN(nn.Module):
    #fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
    #rnn_type = rnn_type, rnn_enc_drop = rnn_enc_drop, rnn_dec_drop = rnn_dec_drop, emb_enc_drop = emb_enc_drop, out_drop = out_drop
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2,
                 rnn_type='GRU', rnn_enc_drop=0.25, rnn_dec_drop=0.1, emb_enc_drop=0.15, out_drop=0.35):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        if rnn_type == 'GRU':
            self.rnn_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=rnn_enc_drop)
        else:
            self.rnn_enc = nn.LSTM(em_sz_enc, nh, num_layers=nl, dropout=rnn_enc_drop)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        if rnn_type == 'GRU':
            self.rnn_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=rnn_dec_drop)
        else:
            self.rnn_dec = nn.LSTM(em_sz_dec, em_sz_dec, num_layers=nl, dropout=rnn_dec_drop)
        self.emb_enc_drop = nn.Dropout(emb_enc_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        
    def forward(self, inp):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        #uses a hook to call .forward(input, hx) on the rnn, where emb = [torch.cuda.FloatTensor of size 35x200x300], h = [torch.cuda.FloatTensor of size 3x200x256]
        enc_out, h = self.rnn_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.rnn_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all(): 
                break
        return torch.stack(res)
    
    def initHidden(self, bs):
        zero_tensor = V(torch.zeros(self.nl, bs, self.nh))
        return zero_tensor

#cross entropy loss. Sequence len of venerated may be different to target (may have stopped early), so need to add some padding
def seq2seq_loss(input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    #our rank 3 tensor input [seq_len x bs x vocab_len] requires a 6 tuple
    if sl>sl_in: 
        input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    #cross entropy loss expects a rank 2 tensor so we do view(-1) to flatten
    return F.cross_entropy(input.view(-1,nc), target.view(-1))#, ignore_index=1)

def run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                        nl=2, rnn_type='GRU', rnn_enc_drop=0.25, rnn_dec_drop=0.1, emb_enc_drop=0.15, out_drop=0.35):
    print('>>run_seq2seq_learner() {0}'.format(rnn_type))
    rnn = Seq2SeqRNN(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, nl=nl,
                    rnn_type=rnn_type, rnn_enc_drop=rnn_enc_drop, rnn_dec_drop=rnn_dec_drop, emb_enc_drop=emb_enc_drop, out_drop=out_drop)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    #learn.lr_find()
    #learn.sched.plot()
    #plt.show(block=False)
    return learn

def run_seq2seq_learn_fit(learn, run_id, lr=3e-3):
    print('>>run_seq2seq_learn_fit()')
    vals_s2s, ep_vals_s2s = learn.fit(lr, n_cycle=1, cycle_len=12, use_clr=(20,10), get_ep_vals=True)
    save_obj(ep_vals_s2s, 'translate_ep_vals_s2s'+'_'+str(run_id))
    learn.save('translate_learn_s2s'+'_'+str(run_id))


def run_seq2seq_preds(learn, val_dl, fr_itos, en_itos, run_id):

    learn.load('initial'+'_'+run_id)

    # ## Test
    x, y = next(iter(val_dl))
    probs = learn.model(V(x))
    probs_arr = learn.predict_array(x)
    preds = to_np(probs.max(2)[1])

    preds[:,1]
    probs_arr[:,1], probs[:,1]

    for i in range(170,173):
        print('x: '+' '.join([fr_itos[o] for o in x[:,i] if o != 1]))
        print('y: '+ ' '.join([en_itos[o] for o in y[:,i] if o != 1]))
        print('pred: '+' '.join([en_itos[o] for o in preds[:,i] if o!=1]))
        print()


# 1:12
# Even sequence to sequence models with limited data can be suprisingly effective

# ## Bdir
# 
# For classification, we took all the tokens, turned them around then trained a new classifier
# 
# People tend not to do bidirectional on Decoder but may work



class Seq2SeqRNN_Bidir(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        #note we specify bidirectional=True
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25, bidirectional=True)
        self.out_enc = nn.Linear(nh*2, em_sz_dec, bias=False)
        self.drop_enc = nn.Dropout(0.05)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        
    def forward(self, inp):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        #only real difference to Seq2SeqRNN above
        h = h.view(2,2,bs,-1).permute(0,2,1,3).contiguous().view(2,bs,-1)
        h = self.out_enc(self.drop_enc(h))

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all(): break
        return torch.stack(res)
    #note now initHidden retuns nl*2
    def initHidden(self, bs): return V(torch.zeros(self.nl*2, bs, self.nh))
        



def run_bdir_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, run_id, lr=3e-3):
    rnn = Seq2SeqRNN_Bidir(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    vals_bdir, ep_vals_bdir = learn.fit(lr, 1, cycle_len=12, use_clr=(20,10), get_ep_vals=True)
    save_obj(ep_vals_bdir, 'translate_ep_vals_bdir'+'_'+run_id)
    learn.save('bidir'+'_'+run_id)


# ## Teacher forcing
# 
# When model starts learning, it doesnt know anything. Early learning will be very difficult.
# 
# So instead of feeding in the thing predicted just now, feed in the actual correct word it was meant to be.
# 
# Cant do at inference time, so we have a pr_force variable. At start of training we set this very high



#override step method of Stepper
class Seq2SeqStepper(Stepper):
    def step(self, xs, y, epoch):
        #this line is the only change to step. Linear decrease to 0 after 10 epochs
        self.m.pr_force = (10-epoch)*0.1 if epoch<10 else 0
        xtra = []
        output = self.m(*xs, y)
        if isinstance(output,tuple): output,*xtra = output
        self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.clip:   # Gradient clipping
            nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        return raw_loss.data[0]




class Seq2SeqRNN_TeacherForcing(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        self.pr_force = 1.
        
    def forward(self, inp, y=None):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all(): 
                break
            #note pr_force, is set high when we start, decrease to zero near end
            if (y is not None) and (random.random()<self.pr_force):
                #if already longer than the target sentence then stop
                if i>=len(y): 
                    break
                dec_inp = y[i]
        return torch.stack(res)
    
    def initHidden(self, bs): return V(torch.zeros(self.nl, bs, self.nh))

def run_teacher_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, run_id, lr=3e-3):
    rnn = Seq2SeqRNN_TeacherForcing(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    vals_force, ep_vals_force = learn.fit(lr, 1, cycle_len=12, use_clr=(20,10), stepper=Seq2SeqStepper, get_ep_vals=True)
    save_obj(ep_vals_force, 'translate_ep_vals_force'+'_'+run_id)
    learn.save('forcing'+'_'+run_id)


# ## Attention Model
# 
# 1:32 Use intermediate state steps rather than just final. Train NN to figure out what is important
# 
# see https://distill.pub/2016/augmented-rnns/

# ![title](img/olah_carter_2016.png)



def rand_t(*sz):
    print(f'sz[0]: {sz[0]}')
    return torch.randn(sz)/math.sqrt(sz[0])

def rand_p(*sz):
    #Paramater is identitical to Variable but tells pytorch to learn the weights
    return nn.Parameter(rand_t(*sz))

class Seq2SeqAttnRNN(nn.Module):
    #fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
    #                     nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2,
                 rnn_type='GRU', rnn_enc_drop=0.25, rnn_dec_drop=0.1, emb_enc_drop=0.15, out_drop=0.35):
        assert nh == 256
        assert nl == 2
        print('Seq2SeqAttnRNN '+ rnn_type)
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        if rnn_type == 'GRU':
            self.rnn_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=rnn_enc_drop)
        else:
            self.rnn_enc = nn.LSTM(em_sz_enc, nh, num_layers=nl, dropout=rnn_enc_drop)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        if rnn_type == 'GRU':
            self.rnn_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=rnn_dec_drop)
        else:
            self.rnn_dec = nn.LSTM(em_sz_dec, em_sz_dec, num_layers=nl, dropout=rnn_dec_drop)
        self.emb_enc_drop = nn.Dropout(emb_enc_drop)
        self.out_drop = nn.Dropout(out_drop)
        self.out = nn.Linear(em_sz_dec*2, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

        #random weight watrix
        self.W1 = rand_p(nh, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec+nh, em_sz_dec)
        self.V = rand_p(em_sz_dec)
        
    def forward(self, inp, y=None, ret_attn=False):
        sl, bs=inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.rnn_enc(emb, h)
        h = self.out_enc(h)
        
        dec_inp = V(torch.zeros(bs).long())
        res, attns = [],[]
        w1e = enc_out @ self.W1
        for i in range(self.out_sl):
            w2h = self.l2(h[-1])
            u = F.tanh(w1e + w2h)
            a = F.softmax(u @ self.V, 0)
            attns.append(a)
            Xa = (a.unsqueeze(2)*enc_out).sum(0)
            emb = self.emb_dec(dec_inp)
            wgt_enc = self.l3(torch.cat([emb, Xa], 1))
            
            outp, h = self.rnn_dec(wgt_enc.unsqueeze(0), h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all():
                break
            if (y is not None) and (random.random()<self.pr_force):
                if i >= len(y): 
                    break
                dec_inp = y[i]
                
        res = torch.stack(res)
        if ret_attn:
            res = res, torch.stack(attns)
        return res
              
    def initHidden(self, bs):
        return V(torch.zeros(self.nl, bs, self.nh))

#md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
#                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3
def run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                       nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3):
    assert nh == 256
    assert nl == 2
    assert len(run_id) > 0
    rnn = Seq2SeqAttnRNN(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                         nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    vals_attn, ep_vals_attn = learn.fit(lr, 1, cycle_len=12, use_clr=(20,10), stepper= Seq2SeqStepper, get_ep_vals=True)
    save_obj(ep_vals_attn, 'translate_ep_vals_attn'+'_'+str(run_id))
    learn.save('translate_learn_attn'+'_'+str(run_id))


def run_attn_preds(learn, val_dl, fr_itos, en_itos):

    learn.load('attn')
    # ## Test
    x,y = next(iter(val_dl))
    probs,attns = learn.model(V(x),ret_attn=True)
    preds = to_np(probs.max(2)[1])


    for i in range(180,190):
        print(' '.join([fr_itos[o] for o in x[:,i] if o != 1]))
        print(' '.join([en_itos[o] for o in y[:,i] if o != 1]))
        print(' '.join([en_itos[o] for o in preds[:,i] if o!=1]))
        print()


def plot_attn(attns):

    #attention for one particular sentence
    attn = to_np(attns[...,180])

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i,ax in enumerate(axes.flat):
        ax.plot(attn[i])


# ## All
class Seq2SeqRNN_All(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25, bidirectional=True)
        self.out_enc = nn.Linear(nh*2, em_sz_dec, bias=False)
        self.drop_enc = nn.Dropout(0.25)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

        self.W1 = rand_p(nh*2, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec+nh*2, em_sz_dec)
        self.V = rand_p(em_sz_dec)

    def forward(self, inp, y=None):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = h.view(2,2,bs,-1).permute(0,2,1,3).contiguous().view(2,bs,-1)
        h = self.out_enc(self.drop_enc(h))

        dec_inp = V(torch.zeros(bs).long())
        res,attns = [],[]
        w1e = enc_out @ self.W1
        for i in range(self.out_sl):
            w2h = self.l2(h[-1])
            u = F.tanh(w1e + w2h)
            a = F.softmax(u @ self.V, 0)
            attns.append(a)
            Xa = (a.unsqueeze(2) * enc_out).sum(0)
            emb = self.emb_dec(dec_inp)
            wgt_enc = self.l3(torch.cat([emb, Xa], 1))
            
            outp, h = self.gru_dec(wgt_enc.unsqueeze(0), h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all(): break
            if (y is not None) and (random.random()<self.pr_force):
                if i>=len(y): break
                dec_inp = y[i]
        return torch.stack(res)

    def initHidden(self, bs): return V(torch.zeros(self.nl*2, bs, self.nh))

def run_all_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, run_id, lr=2e-3):
    rnn = Seq2SeqRNN_All(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss
    vals, ep_vals_all = learn.fit(lr, 1, cycle_len=12, use_clr=(20,10), stepper=Seq2SeqStepper, get_ep_vals = True)
    save_obj(ep_vals_all, 'translate_ep_vals_all')


# ### Compare all the RNN architecture preformances
from matplotlib.pyplot import cm 
def plot_all_ep_vals(ep_val_dict):
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    color=iter(cm.rainbow(np.linspace(0,1,len(ep_val_dict))))
    for k, v in ep_val_dict.items():
        epochs = ep_val_dict[k].keys()
        plt.xticks(np.asarray(list(epochs)))
        val_losses = [item[1] for item in list(ep_val_dict[k].values())]
        c=next(color)
        plt.plot(epochs, val_losses, c=c, label=k)
    plt.yscale('log')
    plt.legend(loc='upper left')

def plot_all_val_loss():
    ep_val_dict = {}
    ep_val_dict['seq2seq'] = load_obj('translate_ep_vals_s2s')
    ep_val_dict['bdir'] = load_obj('translate_ep_vals_bdir')
    ep_val_dict['teacher'] = load_obj('translate_ep_vals_force')
    ep_val_dict['attn'] = load_obj('translate_ep_vals_attn')
    ep_val_dict['all'] = load_obj('translate_ep_vals_all')
    plot_all_ep_vals(ep_val_dict)

def seq2seq_param_sens(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, range_strt, range_stop, rnn_type = 'GRU'):

    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35
    nl=2

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_red_'+str(i)
        rnn_enc_drop = i/10
        learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                                    nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
        run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)

    #reset params
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_rdd_'+str(i)
        rnn_dec_drop = i/10
        learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                                    nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
        run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)

    #reset params
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_eed_'+str(i)
        emb_enc_drop = i/10
        learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                                    nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
        run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)

    #reset params
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_od_'+str(i)
        out_drop = i/10
        learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                                    nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
        run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)

def attn_param_sens(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, range_strt, range_stop, rnn_type = 'GRU'):
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35
    nl = 2

    for i in range(range_strt, range_stop):
        run_id = rnn_type + '_red_'+str(i)
        rnn_enc_drop = i/10
        run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)

    #reset
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type + '_rdd_'+str(i)
        rnn_dec_drop = i/10
        run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)

    #reset
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_eed_'+str(i)
        emb_enc_drop = i/10
        run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)

    #reset
    rnn_enc_drop = 0.25
    rnn_dec_drop = 0.1
    emb_enc_drop = 0.15
    out_drop = 0.35

    for i in range(range_strt, range_stop):
        run_id = rnn_type+'_od_'+str(i)
        out_drop = i/10
        run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)



def run_attn(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                        nl=2, rnn_type='GRU', rnn_enc_drop=0.25, rnn_dec_drop=0.1, emb_enc_drop=0.15, out_drop=0.35):
    #based on plot parameters
    rnn_enc_drop = 0.1
    rnn_dec_drop = 0.2
    emb_enc_drop = 0.4
    out_drop = 0.2
    nl = 2

    run_id = rnn_type + '_plot_params'
    run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                           nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)

def run_attn_drop_0(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99):
    nl = 2
    rnn_type = 'GRU'
    rnn_enc_drop = 0
    rnn_dec_drop = 0
    emb_enc_drop = 0
    out_drop = 0
    range_strt = 1
    range_stop = 10

    #output prefix: 'translate_ep_vals_attn'

    '''
    # Base run all dropouts zero
    run_id = rnn_type + '_all_drop_0'
    run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                       nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)
    '''

    for drop in ['red', 'rdd', 'eed', 'od']:
        for i in range(range_strt, range_stop):
            rnn_enc_drop = 0
            rnn_dec_drop = 0
            emb_enc_drop = 0
            out_drop = 0
            run_id = rnn_type + f'_all_drop_0_{drop}_{i}'
            drop_val = i / 10
            if drop == 'red':
                rnn_enc_drop = drop_val
            elif drop == 'rdd':
                rnn_dec_drop = drop_val
            elif drop=='eed':
                emb_enc_drop = drop_val
            elif drop=='od':
                out_drop = drop_val
            run_attn_learn_fit(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                               nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop, run_id, lr=2e-3)


def run_s2s_drop_0(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99):

    nl=2
    rnn_type='GRU'
    rnn_enc_drop=0
    rnn_dec_drop=0
    emb_enc_drop=0
    out_drop=0
    range_strt=9
    range_stop=10

    #for nl in [2, 3, 4]:
    for nl in [2]:
        run_id = f'GRU_nl_{nl}_all_drop_0'

        #Base run all dropouts zero
        run_id = f'GRU_nl_{nl}_all_drop_0'
        learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99,
                                    nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
        run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)


        for drop in ['red', 'rdd', 'eed', 'od']:
            for i in range(range_strt, range_stop):
                #reset for each run
                rnn_enc_drop = 0
                rnn_dec_drop = 0
                emb_enc_drop = 0
                out_drop = 0
                run_id = rnn_type + f'_nl_{nl}_all_drop_0_{drop}_{i}'
                drop_val = i / 10
                if drop == 'red':
                    rnn_enc_drop = drop_val
                elif drop == 'rdd':
                    rnn_dec_drop = drop_val
                elif drop=='eed':
                    emb_enc_drop = drop_val
                elif drop=='od':
                    out_drop = drop_val
                learn = run_seq2seq_learner(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh,
                                            enlen_99, nl, rnn_type, rnn_enc_drop, rnn_dec_drop, emb_enc_drop, out_drop)
                run_seq2seq_learn_fit(learn, run_id=run_id, lr=3e-3)


def workflow():
    start = timer()

    en_vecd, fr_vecd, dim_en_vec, dim_fr_vec = load_vectors()
    en_ids, en_itos, en_stoi, fr_ids, fr_itos, fr_stoi = get_ids()
    en_ids_tr, fr_ids_tr, enlen_99, frlen_98 = clip_en_fr_ids(en_ids, fr_ids)
    en_trn, fr_trn, en_val, fr_val = create_train_val_sets(en_ids_tr, fr_ids_tr)
    trn_ds, val_ds = create_datasets(fr_trn,en_trn, fr_val,en_val)
    trn_samp, val_samp = sortish_sample_en(en_trn, en_val)

    trn_dl, val_dl, md = create_dataloaders(trn_ds, val_ds, trn_samp, val_samp)

    run_s2s_drop_0(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)
    range_strt = 1
    range_stop=10
    seq2seq_param_sens(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, range_strt,
                       range_stop, rnn_type = 'GRU')
    run_attn_drop_0(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)
    #NB when use LSTM get a RuntimeError: Expected hidden[0] size (2, 200, 256), got (200, 256)
    attn_param_sens(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99, range_strt, range_stop)
    
    '''
    run_attn(md, fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_99)





    '''
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()