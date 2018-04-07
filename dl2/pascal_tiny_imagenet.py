
# coding: utf-8

# ## Lesson8 and 9 pascal object detecion applied to Tiny ImageNet Visual Recognition Challenge
# https://tiny-imagenet.herokuapp.com/
# 
# "*Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images. We have released the training and validation sets with images and annotations. We provide both class labels and bounding boxes as annotations; however, you are asked only to predict the class label of each image without localizing the objects. The test set is released without labels*"
# 

# In[3]:




# In[4]:


from IPython.core.debugger import set_trace

from os import listdir
from os.path import isfile, join
import csv

from pathlib import Path
import json
from matplotlib import patches, patheffects
from scipy.special import expit

from fastai.conv_learner import *
from fastai.dataset import *



# In[5]:


#for pycharm notebooks
plt.ion()


# In[6]:


PATH = Path('/mnt/samsung_1tb/Data/fastai/tiny-imagenet/tiny_imagenet_restructured/')


# In[7]:


os.listdir(PATH)


# Stage 1: re-structure the data layout and file structures 
# 
#     JPEGS: all images in one directory
#     find . -name \*.JPEG -exec cp {} /path/tiny_imagenet_restructured/train/JPEGImages/ \;
#     find . -name \*boxes.txt -exec cp {} /path/tiny_imagenet_restructured/train/Boxes/ \;
#     CSV: create df of filename, category
# 
# Stage 2: read the data in to pascal.ipynb-like dicts
# 
# ie we want 
# 
#     trn_anno: trn_anno[o[IMG_ID]].append((bb,o[CAT_ID]))
#     #trn_anno is a dict of {id, [(bbox, category_id), ...]} where coords are TL, BR numpy consistent x,y coords
# 
#     cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
#     trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
#     trn_ids = [o[ID] for o in trn_j[IMAGES]]

# In[8]:


JPEGS = 'train/JPEGImages'
IMG_PATH = PATH/JPEGS


# In[9]:


df_submission_example  = pd.read_csv(PATH/'yukez.txt', names=['file_name', 'code'], delim_whitespace=True)


# In[10]:


df_submission_example.head(n=2)


# In[11]:


code_words = {}

with open(PATH/'words.txt') as f:
  for row in csv.reader(f, delimiter="\t"):
    #We keep only the first of the descriptions
    code = row[1].split(",")[0]
    #replace whitespaces with underscore
    code = "_".join(code.split()).strip().lower()
    code_words[row[0]] = code


# In[12]:


wnids = [line.strip() for line in open(PATH/'wnids.txt', 'r')]
#eg ['n02124075', 'n04067472', 'n04540053',...]


# In[13]:


#keep only those words from our images
code_words = {k: code_words[k] for k in wnids}
#eg {'n01443537': 'goldfish', 'n01629819': 'European fire salamander',...
#code_words


# In[14]:


cats = dict(enumerate(x for x in code_words.values()))
#cats eg {0: 'Egyptian cat', 1: 'reel',...


# In[15]:


fn_bbox = {}
txt_files = [i for i in os.listdir(PATH/'train/Boxes/') if os.path.splitext(i)[1] == '.txt']

for f in txt_files:
    with open(os.path.join(PATH/'train/Boxes/',f)) as file_object:
        for row in csv.reader(file_object, delimiter="\t"):
            fn_bbox[row[0]] = row[1]+' '+row[2]+' '+row[3]+' '+row[4]


# In[16]:


#note there is only a single bounding box per image
#ie {'n02837789_0.JPEG': '8 8 40 63', 'n02837789_1.JPEG': '32 24 54 39',...
#fn_bbox


# In[17]:


file_id_df = pd.DataFrame({'file_id':wnids})
file_id_df.head(n=2)


# In[18]:


jpeg_filenames = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
jpeg_filenames.sort()
jpeg_fn_df = pd.DataFrame(jpeg_filenames)
trn_fns = list(jpeg_fn_df.to_dict().values())[0]
#eg {0: 'n01443537_0.JPEG', 1: 'n01443537_1.JPEG',...


# In[19]:


#so we can look up by value
inv_trn_fns = {v: k for k, v in trn_fns.items()}
inv_cats = {v: k for k, v in cats.items()}
trn_anno = collections.defaultdict(lambda:[])
for k, v in fn_bbox.items():
    core_code = k.split('_')[0]
    bb = fn_bbox[k].split()
    bb = np.array([bb[0], bb[1], bb[2], bb[3]])
    bb = bb.astype(np.int16)
    cat_id = inv_cats[code_words[core_code]]
    trn_anno[inv_trn_fns[k]].append((bb, cat_id))
#trn_anno
im_a = trn_anno[37000]; im_a


# In[20]:


#conversion between numpy style and cs style
def bb_hw(a): 
    return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])


# In[21]:


im = open_image(IMG_PATH/trn_fns[0])


# In[22]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# In[23]:


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# In[24]:


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


# In[25]:


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


# In[26]:


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


# In[27]:


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)


# In[28]:


def get_lrg(b):
    if not b: 
        raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


# In[29]:


#not necessary as only 1 bbox per image, TODO remove this call
trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}


# In[30]:


b,c = trn_lrg_anno[0]
#b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[0]), figsize=(5,10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)
plt.show()


# In[31]:


b,c = trn_lrg_anno[25001]
#b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[25001]), figsize=(5,10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)
plt.show()


# In[32]:


b


# In[33]:


#create filename, category CSV
(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/lrg.csv'


# In[34]:


fn_cat_dict = {}
for k, v in trn_fns.items():
    core_code = v.split('_')[0]
    cat = code_words[core_code]
    fn_cat_dict[v] = cat


# In[35]:


df = pd.DataFrame(list(fn_cat_dict.items()), columns=['filename', 'category'])
df.head(n=2)
df.to_csv(CSV, index=False)


# In[36]:


df.head(n=2)


# In[37]:


#data check
for item in trn_fns:
    if item not in df['filename']:
        print(f'{item}')


# ### Architecture

# In[38]:


#get av size of first 100 images
image_list = list(trn_fns.values())
widths = []
heights = []
for im in image_list[:100]:
    w,h=Image.open(IMG_PATH/im).size
    widths.append(w)
    heights.append(h)
av_w = sum(widths)/len(widths)
av_h = sum(heights)/len(heights)
print(f'avg width: {av_w}, avg height: {av_h}, max w: {max(widths)}, min w: {min(widths)}, max h: {max(heights)}, min h: {min(heights)}')


# In[39]:


f_model = resnet34
size = 64
bs = 64


# In[40]:


transforms = tfms_from_model(f_model, sz = size, aug_tfms=transforms_side_on, crop_type = CropType.NO)
model_data = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=transforms)


# In[41]:


x,y = next(iter(model_data.val_dl))
show_img(model_data.val_ds.denorm(to_np(x))[0])


# In[42]:


#on first pass y values in data.val_dl.dataset (in model.fit() vals = validate(stepper, data.val_dl, metrics)) were arrays of floats (0.0) instead of integers
#problem was with is_single = np.all(label_arr.sum(axis=1)==1), I re-parsed the labels joining words by underscore and converting to lowercase which fixed this issue


# In[43]:


learn = ConvLearner.pretrained(f_model, model_data, metrics=[accuracy])
learn.opt_fn = optim.Adam


# In[44]:


learn.lr_find()


# In[45]:


learn.sched.plot()


# In[46]:


lr = 1e-3


# In[47]:


learn.fit(lrs=lr, n_cycle=1, cycle_len=3)


# In[48]:


learn.sched.plot_lr()


# In[49]:


lrs = np.array([lr/1000, lr/100, lr])


# In[50]:


learn.freeze_to(-2)


# In[51]:


lrf = learn.lr_find(lrs/1000)


# In[52]:


learn.fit(lrs=lrs/5, n_cycle=1, cycle_len=1)


# In[53]:


learn.unfreeze()


# In[54]:


learn.fit(lrs=lrs/5, n_cycle=1, cycle_len=1)


# In[55]:


learn.save('clas_one')


# In[56]:


learn.load('clas_one')


# In[57]:


x, y = next(iter(model_data.val_dl))
x[0]


# In[58]:


batch_preds = predict_batch(learn.model, x)


# In[59]:


batch_preds


# In[60]:


probs = F.softmax(batch_preds, -1)
probs


# In[61]:


x = to_np(x)
preds = to_np(probs)


# In[62]:


preds = np.argmax(preds, -1)


# In[63]:


#draw the predictions
fig, axes = plt.subplots(3, 4, figsize = (12, 8))
for i,ax in enumerate(axes.flat):
    ima=model_data.val_ds.denorm(x)[i]
    b = model_data.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0, 0), b)
plt.tight_layout()
plt.show()


# ## Bbox Only

# In[65]:


#create a model data object for bboxes
#create a model data object for classses
#combine these into one model


# In[66]:


trn_ids = list(trn_fns.keys())
BB_CSV = PATH/'tmp/bb.csv'


# In[67]:


bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
#want space sep data
bbs = [' '.join(str(p) for p in o) for o in bb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn', 'bbox'])
df.to_csv(BB_CSV, index=False)


# In[68]:


BB_CSV.open().readlines()[:5]


# In[69]:


#define architecture
f_model = resnet34
sz = 64
bs = 64


# In[70]:


#lighting is image-average*b + c
augs = [RandomFlip(), RandomRotate(20), RandomLighting(b=0.1, c=0.1)]
augs


# In[71]:


tfms = tfms_from_model(f_model=f_model, sz=sz, aug_tfms=augs)
#classifiers are categorical, here doing regression
# if continuous=True then label_arr = np.array([np.array(csv_labels[i]).astype(np.float32) for i in fnames])
model_data = ImageClassifierData.from_csv(PATH, JPEGS, csv_fname=BB_CSV, bs=bs, tfms = tfms, continuous=True)


# In[72]:


#draw images and bboxes - note problems with bbox locations - need to transform this in same way as indeprendant var
idx=3
fig, axes = plt.subplots(3, 3, figsize=(9,9))
for i, ax in enumerate(axes.flat):
    x,y = next(iter(model_data.aug_dl))
    ima = model_data.val_ds.denorm(to_np(x))[idx]
    #b = bb_hw(to_np(y[idx]))
    b = to_np(y[idx])
    print(f'b {b}')
    show_img(ima, ax=ax)
    draw_rect(ax, b)


# In[73]:


augs = [RandomFlip(tfm_y=TfmType.COORD),
        RandomRotate(30, tfm_y=TfmType.COORD),
        #I havent set tfm_y for lighting - I assume wont have any impact
        RandomLighting(0.1,0.1)]


# In[74]:


idx=3
fig, axes = plt.subplots(3, 3, figsize=(9,9))
for i, ax in enumerate(axes.flat):
    x,y = next(iter(model_data.aug_dl))
    ima = model_data.val_ds.denorm(to_np(x))[idx]
    b = to_np(y[idx])
    print(f'b {b}')
    show_img(ima, ax=ax)
    draw_rect(ax, b)


# In[75]:


#now we create a model where we rotate 50% of the time 
tfm_y = TfmType.COORD
augs = [RandomFlip(tfm_y=tfm_y),
        #rotate up to 3% 50% of time
        RandomRotate(3, p=0.5, tfm_y=tfm_y),
        RandomLighting(0.05, 0.05, tfm_y=tfm_y)]


# In[76]:


tfms = tfms_from_model(f_model, sz,aug_tfms=augs, crop_type=CropType.NO, tfm_y=tfm_y)
#path, folder, csv_fname, bs=64, tfms
model_data = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, bs=bs, tfms=tfms, continuous=True)


# In[80]:


#create a custom head, 512*7*7 = 25088. 4 bounding box coords
#Dont want to add standard fully connected layers that make up a classifier, want a single linear with 4 outputs
#flattens out, this is a simple final layer. 25088=512x7x7
head_reg4 = nn.Sequential(Flatten(), nn.Linear(in_features=25088, out_features=4))
#custom head wont create any of the fully connected network, wont add the adaptive average pooling, instead will add the model you specify
#note that f_model is resnet34
#stick this on a pretrained model
learn = ConvLearner.pretrained(f_model, data = model_data, custom_head=head_reg4)
learn.opt_fn = optim.Adam
#L1 Loss function minimizes the absolute differences between the estimated values and the target values
learn.crit = nn.L1Loss()


# In[82]:


learn.summary()


# In[83]:


#learn.lr_find()


# In[ ]:


learn.sched.plot()


# In[ ]:


lr = 2e-3


# In[ ]:


learn.fit(lr, n_cycle=2, cycle_len=1, cycle_mult=2)


# In[ ]:


lrs = np.array([lr/100, lr/10, lr])
layer_groups = learn.get_layer_groups()


# In[ ]:


groups_freeze_to = layer_groups[-2:]


# In[ ]:


learn.freeze_to(-2)


# In[ ]:


learn.lr_find(lrs/1000)
learn.sched.plot()


# In[ ]:


learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.freeze_to(-3)


# In[ ]:


learn.fit(lrs, 1, cycle_len=2)


# In[ ]:


learn.save('reg4')
learn.load('reg4')


# In[ ]:


x, y = next(iter(model_data.val_dl))
learn.model.eval()
preds = to_np(learn.model(VV(x)))


# In[ ]:


fig, axes = plt.subplots(3,4, figsize=(12,8))
for i, ax in enumerate(axes.flat):
    ima = model_data.val_ds.denorm(to_np(x))[i]
    b = preds[i]
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
plt.tight_layout()


# ## Single object detection

# In[131]:


trn_ids = list(trn_fns.keys())


# In[132]:


BB_CSV = PATH/'tmp/bb.csv'

bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
#want space sep data
bbs = [' '.join(str(p) for p in o) for o in bb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn', 'bbox'])
df.to_csv(BB_CSV, index=False)

BB_CSV.open().readlines()[:5]


# In[133]:


#bounding box and largest object
f_model = resnet34
sz = 64
bs = 64


# In[134]:


val_idxs = get_cv_idxs(len(trn_fns))


# In[135]:


augs = [RandomFlip(), RandomRotate(20), RandomLighting(b=0.1, c=0.1)]
tfms = tfms_from_model(f_model,sz=sz, aug_tfms=augs, crop_type=CropType.NO, tfm_y=TfmType.COORD)
#BB_CSV contains only filename, bbox coords
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, bs=bs, tfms=tfms, continuous=True, val_idxs=val_idxs)
print(f'{type(md)}, size: {md.sz}')


# In[136]:


tfms2 = tfms_from_model(f_model, sz=sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
#note we are reading in the CSV not BB_CSV, CSV contains only filename, category
md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, bs=bs, tfms=tfms2)
md2.sz


# In[137]:


class ConcatLblDataset(Dataset):
    #take existing ds which has existing independent and dependant vars, add in dependent y2
    #extends torch Dataset, which requires one to override the methods below
    #sz property added as is required by learner (lean.summary())
    def __init__(self, ds, ds2):
        #I changed this to use seconds dataset rater than just y to help me understand this
        self.ds = ds
        self.ds2 = ds2
        
    def __len__(self):
        assert len(self.ds) == len(self.ds2)
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        x2, y2 = self.ds2[i]
        #these should contain exactly the same info, so we only need to keep one. However this test fails
        #assert np.array_equal(x,x2)
        return (x, (y, y2))
    
    @property
    def sz(self):
        x, y = self.ds[0]
        print(f'sz: {x.shape[1]}')
        return x.shape[1]


# In[138]:


#the bbox tagets
md.trn_ds.y


# In[139]:


#classification dependent variable
md2.trn_ds.y


# In[140]:


#NB I changed the ConcatLblDataset. now it takes two datasets
trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_ds)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_ds)


# In[141]:


val_ds2[0][1]


# In[142]:


md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2
print(f'{type(md)}')


# In[143]:


x, y = next(iter(md.val_dl))
idx=50
ima = md.val_ds.ds.denorm(to_np(x))[idx]
#b = bb_hw(to_np(y[0][idx])); b
b = to_np(y[0][idx]); b


# In[144]:


ax = show_img(ima)
draw_rect(ax,b)
draw_text(ax, b[:2], md2.classes[y[1][idx]])
plt.show()


# In[145]:


#replace head, add an extra linear layer and add dropout
head_reg4 = nn.Sequential(
    Flatten(), 
    nn.ReLU(),
    nn.Dropout(0.5),
    #this is a helper linear layer
    nn.Linear(in_features=512*7*7, out_features=256),
    nn.ReLU(),
    #1d as we have flattened 
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    #note at this stage we only have 256 in_features
    nn.Linear(in_features=256, out_features= 4+len(cats))
)


# In[146]:


#ConvnetBuilder(f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None)
model_arch = ConvnetBuilder(f=f_model, c=0, is_multi=False, is_reg=False, custom_head=head_reg4)


# In[147]:


learn = ConvLearner(data = md, models = model_arch)


# In[148]:


learn.opt_fn = optim.Adam


# In[149]:


def detn_loss(input, target):
    bb_t, c_t = target
    #first dim is batch dim, input first 4, and 4 onwards elements
    bb_i, c_i = input[:, :4], input[:, 4:]
    #for bb_i we know they will be b/w 0 & 64
    #use sigmoid to force b/w 0&1 and force the range
    bb_i = F.sigmoid(bb_i)*64
    #scalar for classifications based on what works
    combined_loss = F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20
    return combined_loss
    
def detn_l1(input, target):
    bb_t, _ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i)*64
    f1_loss = F.l1_loss(V(bb_i), V(bb_t)).data
    return f1_loss

def detn_acc(input, target):
    _,c_t = target
    c_i = input[:,4:]
    return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]


# In[85]:


'''#RuntimeError: size mismatch at
#model.py
#in fit() function
#    stepper = stepper(model, opt, crit, **kwargs)
#when step is called
#class Stepper():
#    def step(self, xs, y, epoch):
#           xtra = []
#>>         output = self.m(*xs)
#where xs is V(x)
#error thrown during forward pass of the Flatten layer at end of 
Conv2d
BatchNorm2d
ReLU
Maxpool2d
Sequential Conv2d (64,64)
Sequential Conv2d(125,256)
Sequential Conv2d(256,512)
>>Flatten inFeatures=25088, outFeatures=256

y at input to step
y[0] size [64,4]
y[1] size [64]
xs at input to step
size [64,3,64,64] cf [64,3,224,224] for same stage in pascal.ipynb
<class 'list'>: [Variable containing:
(0 ,0 ,.,.) = 
 -2.1179 -2.1179 -2.1179  ...  -2.1179 -2.1179 -2.0807
 -2.1179 -2.1179 -2.1179  ...  -2.1179 -2.1179 -2.0893
 -2.1179 -2.1179 -2.1179  ...  -2.1179 -2.1179 -2.1179
           ...             ⋱             ...          
 -1.8174 -1.6263 -1.9091  ...  -1.7022 -1.8412 -2.1179
 -1.7784 -1.8500 -1.8215  ...  -1.7622 -1.7095 -1.8014
 -1.7473 -1.9146 -1.7798  ...  -1.9395 -1.7723 -1.3995

(0 ,1 ,.,.) = 
 -1.0885 -1.2341 -1.2682  ...  -1.6536 -1.6536 -1.6346
 -1.0888 -1.2725 -1.2161  ...  -1.6589 -1.6642 -1.6434
 -1.0437 -1.1652 -1.0592  ...  -1.7392 -1.7629 -1.7362
           ...             ⋱             ...          
  0.1706  0.3583  0.0657  ...  -0.4843 -0.7044 -1.3552
  0.1845  0.1092  0.1493  ...  -0.5185 -0.5477 -0.7424
  0.1847  0.0080  0.1636  ...  -0.6836 -0.5709 -0.2614

'''
#learn.lr_find()
#learn.sched.plot()


# In[ ]:


lr=1e-3
#use_clr sets shed to use CircularLR
learn.fijupyter lab --port 8889
t(lrs=lr, n_cycle=1, cycle_len=3, use_clr=(32,5))


# In[ ]:


learn.save('reg1_0')


# In[ ]:


layer_groups = learn.get_layer_groups()
print(f'{layer_groups}, {len(layer_groups)}')


# In[ ]:


#set to trainable the last two layer groups only 
learn.freeze_to(-2) 


# In[ ]:


lrs = np.array([lr/100, lr/10, lr])


# In[ ]:


learn.lr_find(lrs/1000)
learn.sched.plot(0)


# In[ ]:


learn.fit(lrs/5, n_cycle=1, cycle_len=5, use_clr=(32,10))


# In[ ]:


learn.save('reg1_1')


# In[ ]:


learn.load('reg1_1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(lrs=lrs/10, n_cycle=1, cycle_len=10, use_clr=(32, 10))


# In[ ]:


learn.save('reg1')


# In[ ]:


learn.load('reg1')


# In[ ]:


preds = learn.predict()
x, _ = next(iter(md.val_dl))


# In[ ]:


fig, axes = plt.subplots(3, 4, figsize=(12,8))
for i, ax in enumerate(axes.flat):
    #val_ds is val_dl.dataset but val_ds.ds?, 
    #.denorm is a method in FilesDataset where we reverse the normalization done to a batch of images
    ima = md.val_ds.ds.denorm(to_np(x))[i]
    #module scipy.special._ufuncs: The expit function, also known as the logistic function, is defined as
    #expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function
    print(f'preds[i][:4]: {preds[i][:4]}')
    bb = expit(preds[i][:4])*64
    print(f'bb: {bb}')
    #b = bb_hw(bb)
    c = np.argmax(preds[i][4:])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, bb)
    draw_text(ax, bb[:2], md2.classes[c])
plt.tight_layout()

