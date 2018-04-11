
# coding: utf-8

# ## Lesson8 and 9 pascal object detecion applied to Tiny ImageNet Visual Recognition Challenge
# https://tiny-imagenet.herokuapp.com/
# 
# "*Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images. We have released the training and validation sets with images and annotations. We provide both class labels and bounding boxes as annotations; however, you are asked only to predict the class label of each image without localizing the objects. The test set is released without labels*"
# 

# In[1]:




# In[61]:


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



# In[3]:


#for pycharm notebooks
plt.ion()


# In[11]:

PATH = Path('..')/'data/tiny-imagenet/tiny_imagenet_restructured/'


# In[12]:


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

# In[13]:


JPEGS = 'train/JPEGImages'
IMG_PATH = PATH/JPEGS


# In[44]:


df_submission_example  = pd.read_csv(PATH/'yukez.txt', names=['file_name', 'code'], delim_whitespace=True)


# In[45]:


df_submission_example.head(n=2)


# In[51]:


code_words = {}

with open(PATH/'words.txt') as f:
  for row in csv.reader(f, delimiter="\t"):
    #We keep only the first of the descriptions
    code_words[row[0]] = row[1].split(",")[0]


# In[52]:


wnids = [line.strip() for line in open(PATH/'wnids.txt', 'r')]
#eg ['n02124075', 'n04067472', 'n04540053',...]


# In[55]:


#keep only those words from our images
code_words = {k: code_words[k] for k in wnids}
#eg {'n01443537': 'goldfish', 'n01629819': 'European fire salamander',...
#code_words


# In[81]:


cats = dict(enumerate(x.strip() for x in code_words.values()))
#cats eg {0: 'Egyptian cat', 1: 'reel',...


# In[73]:


fn_bbox = {}
txt_files = [i for i in os.listdir(PATH/'train/Boxes/') if os.path.splitext(i)[1] == '.txt']

for f in txt_files:
    with open(os.path.join(PATH/'train/Boxes/',f)) as file_object:
        for row in csv.reader(file_object, delimiter="\t"):
            fn_bbox[row[0]] = row[1]+' '+row[2]+' '+row[3]+' '+row[4]


# In[75]:


#note there is only a single bounding box per image
#ie {'n02837789_0.JPEG': '8 8 40 63', 'n02837789_1.JPEG': '32 24 54 39',...
#fn_bbox


# In[59]:


file_id_df = pd.DataFrame({'file_id':wnids})
file_id_df.head(n=2)


# In[71]:


jpeg_filenames = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
jpeg_filenames.sort()
jpeg_fn_df = pd.DataFrame(jpeg_filenames)
trn_fns = list(jpeg_fn_df.to_dict().values())[0]
#eg {0: 'n01443537_0.JPEG', 1: 'n01443537_1.JPEG',...


# In[133]:


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


# In[134]:


#conversion between numpy style and cs style
def bb_hw(a): 
    return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])


# In[135]:


im = open_image(IMG_PATH/trn_fns[0])


# In[136]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# In[137]:


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# In[138]:


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


# In[139]:


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


# In[140]:


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


# In[141]:


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)


# In[142]:


def get_lrg(b):
    if not b: 
        raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


# In[143]:


#not necessary as only 1 bbox per image, TODO remove this call
trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}


# In[144]:


b,c = trn_lrg_anno[0]
b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[0]), figsize=(5,10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)
plt.show()


# In[145]:


#create filename, category CSV
(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/lrg.csv'


# In[146]:


fn_cat_dict = {}
for k, v in trn_fns.items():
    core_code = v.split('_')[0]
    cat = code_words[core_code]
    fn_cat_dict[v] = cat


# In[147]:


df = pd.DataFrame(list(fn_cat_dict.items()), columns=['filename', 'category'])
df.head(n=2)
df.to_csv(CSV, index=False)


# In[148]:


df.head(n=2)


# ### Architecture

# In[149]:


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


# In[150]:


f_model = resnet34
size = 64
bs = 64


# In[151]:


transforms = tfms_from_model(f_model, sz = size, aug_tfms=transforms_side_on, crop_type = CropType.NO)
model_data = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=transforms)


# In[152]:


x,y = next(iter(model_data.val_dl))
show_img(model_data.val_ds.denorm(to_np(x))[0])

print(y)
# In[153]:


learn = ConvLearner.pretrained(f_model, model_data, metrics=[accuracy])
learn.opt_fn = optim.Adam


# In[154]:


learn.lr_find()


# In[155]:


learn.sched.plot()


# In[156]:


lr = 1e-3


# In[157]:


learn.fit(lrs=lr, n_cycle=1, cycle_len=3)


# In[ ]:


learn.sched.plot_lr()

