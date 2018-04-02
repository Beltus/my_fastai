
# coding: utf-8

# In[ ]:


#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline

from fastai.conv_learner import *
from fastai.plots import *
from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *

from sklearn.metrics import fbeta_score
import warnings

plt.ion()

PATH = '/mnt/samsung_1tb/Data/fastai/planet/'

def get_1st(path):
    return glob(f'{path}/*.*')[0]


# In[ ]:


list_paths = [f'{PATH}train-jpg/train_0.jpg', f'{PATH}train-jpg/train_1.jpg']
titles = ['hazt primary', 'agriculture clear primary water']


# In[ ]:


#from planet.py
def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

def opt_th(preds, targs, start=0.17, end=0.24, step=0.01):
    ths = np.arange(start,end,step)
    idx = np.argmax([fbeta_score(targs, (preds>th), 2, average='samples')
                for th in ths])
    return ths[idx]

def get_data(path, tfms, bs, n, cv_idx):
    val_idxs = get_cv_idxs(n, cv_idx)
    return ImageClassifierData.from_csv(path, 'train-jpg', f'{path}train_v2.csv', bs, tfms,
                                 suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')

def get_data_zoom(f_model, path, sz, bs, n, cv_idx):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return get_data(path, tfms, bs, n, cv_idx)

def get_data_pad(f_model, path, sz, bs, n, cv_idx):
    transforms_pt = [RandomRotateZoom(9, 0.18, 0.1), RandomLighting(0.05, 0.1), RandomDihedral()]
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_pt, pad=sz//12)
    return get_data(path, tfms, bs, n, cv_idx)


# In[ ]:


metrics = [f2]
f_model = resnet34

label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)

def local_get_data(sz, bs=64):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    data = ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, bs=bs, tfms=tfms, suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')
    return data


# In[ ]:


data = local_get_data(sz=256)

x,y = next(iter(data.val_dl))

learn = ConvLearner.pretrained(f_model,data,metrics=metrics)


# In[ ]:


lr = 0.2

lrs = np.array([lr/9, lr/3, lr])


# In[ ]:


sz=256

#drop batch size to fix cude out of memory error
bs = 48

learn.set_data(local_get_data(sz, bs))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.unfreeze()
#cuda out of memory error on 1080ti using sz=256, bs=64
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[ ]:


multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)
f2(preds, y)

val = learn.predict()

f2(val, data.val_y)

##
f2(learn.TTA(), data.val_y)

#ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

f2(val, data.val_y)

f2(learn.TTA(), data.val_y)

def get_labels(a): return [data.classes[o] for o in a.nonzero()[0]]


# In[ ]:


lbls = test>0.2
idx=print(get_labels(lbls[idx]))
PIL.Image.open(PATH+data.test_dl.dataset.fnames[idx]).convert('RGB')


# In[ ]:


res = [get_labels(o) for o in lbls]
data.test_dl.dataset.fnames[:5]


# In[ ]:


outp = pd.DataFrame({'image_name': [f[9:-4] for f in data.test_dl.dataset.fnames], 'tags': [' '.join(l) for l in res]})
outp.head()

outp.to_csv('tmp/subm.gz', compression='gzip', index=None)

