
# coding: utf-8

# In[98]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[99]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(0)


# ## Pascal VOC

# We will be looking at the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. It's quite slow, so you may prefer to download from [this mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). There are two different competition/research datasets, from 2007 and 2012. We'll be using the 2007 version. You can use the larger 2012 for better results, or even combine them (but be careful to avoid data leakage between the validation sets if you do this).
# 
# Unlike previous lessons, we are using the python 3 standard library `pathlib` for our paths and file access. Note that it returns an OS-specific class (on Linux, `PosixPath`) so your output may look a little different. Most libraries than take paths as input can take a pathlib object - although some (like `cv2`) can't, in which case you can use `str()` to convert it to a string.

# In[100]:


PATH = Path('/mnt/samsung_1tb/Data/fastai/pascal/pascal_direct')
list(PATH.iterdir())


# In[101]:


PATH.iterdir()


# As well as the images, there are also *annotations* - *bounding boxes* showing where each object is. These were hand labeled. The original version were in XML, which is a little hard to work with nowadays, so we uses the more recent JSON version which you can download from [this link](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip).
# 
# You can see here how `pathlib` includes the ability to open files (amongst many other capabilities).

# In[102]:


trn_j = json.load((PATH / 'pascal_train2007.json').open())
trn_j.keys()


# In[103]:


IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']
trn_j[IMAGES][:5]


# In[104]:


trn_j[ANNOTATIONS][:2]


# In[105]:


trn_j[CATEGORIES][:4]


# It's helpful to use constants instead of strings, since we get tab-completion and don't mistype.

# In[106]:


FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'

cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
trn_ids = [o[ID] for o in trn_j[IMAGES]]


# In[107]:


(PATH/'train'/'VOC2007').iterdir()


# In[108]:


list((PATH/'train'/'VOC2007').iterdir())


# In[109]:


JPEGS = 'train/VOC2007/JPEGImages'


# In[110]:


IMG_PATH = PATH/JPEGS
list(IMG_PATH.iterdir())[:5]


# Each image has a unique ID.

# In[111]:


im0_d = trn_j[IMAGES][0]
im0_d[FILE_NAME],im0_d[ID]


# A `defaultdict` is useful any time you want to have a default dictionary entry for new keys. Here we create a dict from image IDs to a list of annotations (tuple of bounding box and class id).
# 
# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.

# In[112]:


trn_anno = collections.defaultdict(lambda:[])
for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
        trn_anno[o[IMG_ID]].append((bb,o[CAT_ID]))
        
len(trn_anno)


# In[113]:


im_a = trn_anno[im0_d[ID]]; im_a


# In[114]:


im_0a = im_a[0]


# In[115]:


im_0a


# In[116]:


cats[7]


# In[117]:


trn_anno[17]


# In[118]:


cats[15],cats[13]


# Some libs take VOC format bounding boxes, so this let's us convert back when required:

# In[119]:


def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])


# You can use [Visual Studio Code](https://code.visualstudio.com/) (vscode - open source editor that comes with recent versions of Anaconda, or can be installed separately), or most editors and IDEs, to find out all about the `open_image` function. vscode things to know:
# 
# - Command palette (<kbd>Ctrl-shift-p</kbd>)
# - Select interpreter (for fastai env)
# - Select terminal shell
# - Go to symbol (<kbd>Ctrl-t</kbd>)
# - Find references (<kbd>Shift-F2</kbd>)
# - Go to definition (<kbd>F12</kbd>)
# - Go back (<kbd>alt-left</kbd>)
# - View documentation
# - Hide sidebar (<kbd>Ctrl-b</kbd>)
# - Zen mode (<kbd>Ctrl-k,z</kbd>)

# In[120]:


im = open_image(IMG_PATH/im0_d[FILE_NAME])


# Matplotlib's `plt.subplots` is a really useful wrapper for creating plots, regardless of whether you have more than one subplot. Note that Matplotlib has an optional object-oriented API which I think is much easier to understand and use (although few examples online use it!)

# In[121]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# A simple but rarely used trick to making text visible regardless of background is to use white text with black outline, or visa versa. Here's how to do it in matplotlib.

# In[122]:


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# Note that `*` in argument lists is the [splat operator](https://stackoverflow.com/questions/5239856/foggy-on-asterisk-in-python). In this case it's a little shortcut compared to writing out `b[-2],b[-1]`.

# In[123]:


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


# In[124]:


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


# In[125]:


ax = show_img(im)
b = bb_hw(im_0a[0])
draw_rect(ax, b)
draw_text(ax, b[:2], cats[im_0a[1]])
plt.show()


# In[126]:


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


# In[127]:


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)


# In[128]:


draw_idx(17)


# ## Largest item classifier

# In[129]:


def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


# In[130]:


trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}


# In[131]:


b,c = trn_lrg_anno[23]
b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[23]), figsize=(5,10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)
plt.show()


# In[132]:


(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/lrg.csv'


# In[133]:


df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids],
    'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn','cat'])
df.to_csv(CSV, index=False)


# In[134]:


f_model = resnet34
sz=224
bs=64


# In[135]:


tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms)


# In[136]:


md.trn_dl


# In[137]:


x,y=next(iter(md.val_dl))
show_img(md.val_ds.denorm(to_np(x))[0]);


# In[138]:


learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam


# In[139]:


lrf=learn.lr_find(1e-5,100)


# In[140]:


learn.sched.plot()
plt.show()


# In[141]:


learn.sched.plot(n_skip=5, n_skip_end=1)
plt.show()


# In[142]:


lr = 2e-2


# In[143]:


learn.fit(lr, 1, cycle_len=1)


# In[144]:


lrs = np.array([lr/1000,lr/100,lr])


# In[145]:


learn.freeze_to(-2)


# In[146]:


lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)


# In[147]:


learn.fit(lrs/5, 1, cycle_len=1)


# In[148]:


learn.unfreeze()


# In[149]:


learn.fit(lrs/5, 1, cycle_len=2)


# In[150]:


learn.save('clas_one')


# In[151]:


learn.load('clas_one')


# In[152]:


x,y = next(iter(md.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x,preds = to_np(x),to_np(probs)
preds = np.argmax(preds, -1)


# You can use the python debugger `pdb` to step through code.
# 
# - `pdb.set_trace()` to set a breakpoint
# - `%debug` magic to trace an error
# 
# Commands you need to know:
# 
# - s / n / c
# - u / d
# - p
# - l

# In[153]:


i=0


# In[154]:


ima=md.val_ds.denorm(x)[i]


# In[155]:


b = md.classes[preds[i]]


# In[156]:


ax = show_img(ima, ax=ax)


# In[157]:


draw_text(ax, (0,0), b)


# In[158]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    #pdb.set_trace()
    ima=md.val_ds.denorm(x)[i]
    b = md.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()
plt.show()


# In[87]:


#get_ipython().run_line_magic('debug', '')


# ## Bbox only

# In[89]:


BB_CSV = PATH/'tmp/bb.csv'


# In[90]:


bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
bbs = [' '.join(str(p) for p in o) for o in bb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn','bbox'])
df.to_csv(BB_CSV, index=False)


# In[91]:


f_model=resnet34
sz=224
bs=64


# In[92]:


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True)#, num_workers=0)


# In[93]:


x,y=next(iter(md.val_dl))


# In[94]:


ima=md.val_ds.denorm(to_np(x))[0]
b = bb_hw(to_np(y[0])); b


# In[95]:


ax = show_img(ima)
draw_rect(ax, b)
draw_text(ax, b[:2], 'label')


# In[88]:


head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088,4))
learn = ConvLearner.pretrained(f_model, md, custom_head=head_reg4)
learn.opt_fn = optim.Adam
learn.crit = nn.L1Loss()


# In[147]:


learn.lr_find(1e-5,100)
learn.sched.plot(5)
plt.show()


# In[151]:


lr = 2e-3


# In[152]:


learn.fit(lr, 2, cycle_len=1, cycle_mult=2)


# In[153]:


lrs = np.array([lr/100,lr/10,lr])


# In[154]:


learn.freeze_to(-2)


# In[106]:


lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)
plt.show()


# In[155]:


learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)


# In[156]:


learn.freeze_to(-3)


# In[157]:


learn.fit(lrs, 1, cycle_len=2)


# In[158]:


learn.save('reg4')


# In[89]:


learn.load('reg4')


# In[97]:


x,y = next(iter(md.val_dl))
learn.model.eval()
preds = to_np(learn.model(VV(x)))


# In[99]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.denorm(to_np(x))[i]
    b = bb_hw(preds[i])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
plt.tight_layout()
plt.show()


# ## Single object detection

# In[100]:


f_model=resnet34
sz=224
bs=64

val_idxs = get_cv_idxs(len(trn_fns))


# In[101]:


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
    continuous=True, num_workers=4, val_idxs=val_idxs)


# In[102]:


md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))


# In[103]:


class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): self.ds,self.y2 = ds,y2
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))


# In[104]:


trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2


# In[166]:


x,y=next(iter(md.val_dl))


# In[167]:


ima=md.val_ds.ds.denorm(to_np(x))[1]
b = bb_hw(to_np(y[0][1])); b


# In[168]:


ax = show_img(ima)
draw_rect(ax, b)
draw_text(ax, b[:2], md2.classes[y[1][1]])


# In[105]:


head_reg4 = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088,256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256,4+len(cats)),
)
models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam


# In[106]:


def detn_loss(input, target):
    bb_t,c_t = target
    bb_i,c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i)*224
    return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20

def detn_l1(input, target):
    bb_t,_ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i)*224
    return F.l1_loss(V(bb_i),V(bb_t)).data

def detn_acc(input, target):
    _,c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]


# In[171]:


learn.lr_find()
learn.sched.plot()
plt.show()


# In[107]:


lr=1e-2


# In[193]:


learn.fit(lr, 1, cycle_len=3, use_clr=(32,5))


# In[194]:


learn.save('reg1_0')


# In[195]:


learn.freeze_to(-2)


# In[196]:


lrs = np.array([lr/100, lr/10, lr])


# In[185]:


learn.lr_find(lrs/1000)
learn.sched.plot(0)


# In[197]:


learn.fit(lrs/5, 1, cycle_len=5, use_clr=(32,10))


# In[198]:


learn.save('reg1_1')


# In[353]:


learn.load('reg1_1')


# In[199]:


learn.unfreeze()


# In[200]:


learn.fit(lrs/10, 1, cycle_len=10, use_clr=(32,10))


# In[201]:


learn.save('reg1')


# In[108]:


learn.load('reg1')


# In[109]:


y = learn.predict()
x,_ = next(iter(md.val_dl))


# In[110]:


from scipy.special import expit


# In[112]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.ds.denorm(to_np(x))[i]
    bb = expit(y[i][:4])*224
    b = bb_hw(bb)
    c = np.argmax(y[i][4:])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
    draw_text(ax, b[:2], md2.classes[c])
plt.tight_layout()
plt.show()


# ## End
