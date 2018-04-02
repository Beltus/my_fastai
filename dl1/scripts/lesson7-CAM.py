
# coding: utf-8

# ## Dogs v Cats

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *


# In[2]:


PATH = "data/dogscats/"
sz = 224
arch = resnet34
bs = 64


# In[3]:


m = arch(True)


# In[4]:


m


# In[5]:


m = nn.Sequential(*children(m)[:-2], 
                  nn.Conv2d(512, 2, 3, padding=1), 
                  nn.AdaptiveAvgPool2d(1), Flatten(), 
                  nn.LogSoftmax())


# In[6]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)


# In[7]:


learn = ConvLearner.from_model_data(m, data)


# In[8]:


learn.freeze_to(-4)


# In[9]:


m[-1].trainable


# In[10]:


m[-4].trainable


# In[11]:


learn.fit(0.01, 1)


# In[12]:


learn.fit(0.01, 1, cycle_len=1)


# ## CAM

# In[13]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = to_np(output)
    def remove(self): self.hook.remove()


# In[14]:


x,y = next(iter(data.val_dl))
x,y = x[None,1], y[None,1]

vx = Variable(x.cuda(), requires_grad=True)


# In[39]:


dx = data.val_ds.denorm(x)[0]
plt.imshow(dx);


# In[15]:


sf = SaveFeatures(m[-4])
py = m(Variable(x.cuda()))
sf.remove()

py = np.exp(to_np(py)[0]); py


# In[16]:


feat = np.maximum(0, sf.features[0])
feat.shape


# In[23]:


f2=np.dot(np.rollaxis(feat,0,3), py)
f2-=f2.min()
f2/=f2.max()
f2


# In[22]:


plt.imshow(dx)
plt.imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='hot');


# ## Model

# In[38]:


learn.unfreeze()
learn.bn_freeze(True)


# In[39]:


lr=np.array([1e-6,1e-4,1e-2])


# In[40]:


learn.fit(lr, 2, cycle_len=1)


# In[41]:


accuracy(*learn.TTA())


# In[42]:


learn.fit(lr, 2, cycle_len=1)


# In[43]:


accuracy(*learn.TTA())

