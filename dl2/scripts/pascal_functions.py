
# coding: utf-8
import logging

from scipy.special import expit

from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(0)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ## Pascal VOC

# We will be looking at the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. It's quite slow, so you may prefer to download from [this mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). There are two different competition/research datasets, from 2007 and 2012. We'll be using the 2007 version. You can use the larger 2012 for better results, or even combine them (but be careful to avoid data leakage between the validation sets if you do this).
# 
# Unlike previous lessons, we are using the python 3 standard library `pathlib` for our paths and file access. Note that it returns an OS-specific class (on Linux, `PosixPath`) so your output may look a little different. Most libraries than take paths as input can take a pathlib object - although some (like `cv2`) can't, in which case you can use `str()` to convert it to a string.

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

PATH = Path('/mnt/samsung_1tb/Data/fastai/pascal/pascal_direct')
logger.debug(f'PATH: {list(PATH.iterdir())}')


# As well as the images, there are also *annotations* - *bounding boxes* showing where each object is. These were hand labeled. The original version were in XML, which is a little hard to work with nowadays, so we uses the more recent JSON version which you can download from [this link](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip).
# 
# You can see here how `pathlib` includes the ability to open files (amongst many other capabilities).

trn_j = json.load((PATH / 'pascal_train2007.json').open())

logger.debug(f'trn_j keys: {trn_j.keys()}')

IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']

logger.debug(f'first 5 images: {trn_j[IMAGES][:5]}')
logger.debug(f'first 2 annotations: {trn_j[ANNOTATIONS][:2]}')
logger.debug(f'first 4 categories: {trn_j[CATEGORIES][:4]}')

# It's helpful to use constants instead of strings, since we get tab-completion and don't mistype.
FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name','id','image_id','category_id','bbox'

cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
trn_ids = [o[ID] for o in trn_j[IMAGES]]

logger.debug(f"VOC2007: {(PATH/'train'/'VOC2007').iterdir()}")

list((PATH/'train'/'VOC2007').iterdir())


JPEGS = 'train/VOC2007/JPEGImages'

IMG_PATH = PATH/JPEGS
logger.debug(f'first 5 IMG_PATH: {list(IMG_PATH.iterdir())[:5]}')


# Each image has a unique ID.
im0_d = trn_j[IMAGES][0]
logger.debug(f'im0_d: {im0_d[FILE_NAME],im0_d[ID]}')

(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/lrg.csv'

# ## Bbox only
BB_CSV = PATH/'tmp/bb.csv'

# A `defaultdict` is useful any time you want to have a default dictionary entry for new keys. Here we create a dict from image IDs to a list of annotations (tuple of bounding box and class id).
# 
# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.


#if key not found in dict, return an empty list
trn_anno = collections.defaultdict(lambda:[])
for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
        # append the values tuple: bbox and category_id to the image_id key
        #if doesnt exist, appends empty list that can add to
        trn_anno[o[IMG_ID]].append((bb,o[CAT_ID]))

logger.debug(f'annotations: {len(trn_anno)}')

def examples():
    logger.debug('>>examples()')
    im_a = trn_anno[im0_d[ID]]
    logger.debug(im_a)

    im_0a = im_a[0]
    logger.debug(im_0a)

    logger.debug(f'cats 7: {cats[7]}')

    logger.debug(f'trn_anno 17: {trn_anno[17]}')

    logger.debug(f'cats 15: {cats[15]}, cats 13: {cats[13]}')
    return im_0a


# Some libs take VOC format bounding boxes, so this let's us convert back when required:
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])


#using open cv - multi threads
im = open_image(IMG_PATH/im0_d[FILE_NAME])


# Matplotlib's `plt.subplots` is a really useful wrapper for creating plots, regardless of whether you have more than one subplot. Note that Matplotlib has an optional object-oriented API which I think is much easier to understand and use (although few examples online use it!)

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #returns axis - created or from inout param
    return ax


# A simple but rarely used trick to making text visible regardless of background is to use white text with black outline, or visa versa. Here's how to do it in matplotlib.

def draw_outline(o, lw):
    #o is mpl plotting object, put back line underneath to can see bb's
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# Note that `*` in argument lists is the [splat operator](https://stackoverflow.com/questions/5239856/foggy-on-asterisk-in-python). In this case it's a little shortcut compared to writing out `b[-2],b[-1]`.
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)

def show_ex(im_0a):
    ax = show_img(im)
    b = bb_hw(im_0a[0])
    draw_rect(ax, b)
    #where b[:2] = 1st 2 vals ie top Left corner coords
    draw_text(ax, b[:2], cats[im_0a[1]])
    plt.show()

def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        #turn into to height and width, then draw
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)

def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)

def draw_ex_n(n):
    draw_idx(n)
    plt.show()


# ## Largest item classifier
def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]

def get_trn_lrg_anno():
    #go through each bbox and get largest - JH wrote this first before the get_lrg function
    #prod of last 2 items of bb list ie br - top left, take product = size
    #dict comprehension {key (imageid): value (largest bb)}
    trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}
    return trn_lrg_anno

def draw_anno_n(trn_lrg_anno, n):
    #look at example
    b,c = trn_lrg_anno[n]
    b = bb_hw(b)
    ax = show_img(open_image(IMG_PATH/trn_fns[n]), figsize=(5,10))
    draw_rect(ax, b)
    draw_text(ax, b[:2], cats[c], sz=16)
    plt.show()


def save_annotations(trn_lrg_anno):
    #col order matters, - dict doesnt have order
    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids],
        'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn','cat'])
    df.to_csv(CSV, index=False)

def create_model():
    logger.debug('>>create_model()')
    f_model = resnet34
    sz=224
    bs=64

    #default for 224x224 is resize to smallest is 224, and random square crop, during val centre crop. But dont want to do for bb's. Squishing images to square
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)

    #simple to use CSV's
    md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms)

    #md is a fastai DataLoader, need to use next(iter(md.trn_dl)) if want to go through
    #grab minibatch
    x,y=next(iter(md.val_dl))
    print(y)
    #cant send straight to show_img as is as x is floattensor on gpu and shape is wrong, plus no's are not b/w 0&1
    #denormalize depends on transform - dataset stores this, pass a numpy minibatch (x)
    show_img(md.val_ds.denorm(to_np(x))[0])
    plt.show()
    return f_model, md


def find_model_lr(f_model, md):
    #note how the lr_find plot is a little different each time it is run
    logger.debug('>>find_model_lr')
    learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
    learn.opt_fn = optim.Adam

    lrf=learn.lr_find(1e-5,100)

    learn.sched.plot()
    plt.show()

    learn.sched.plot(n_skip=5, n_skip_end=1)
    plt.show()
    return learn

def train_model(learn):
    logger.debug('>>train_model')
    lr = 2e-2

    vals = learn.fit(lr, 1, cycle_len=1)
    logger.debug(vals)

    lrs = np.array([lr/1000,lr/100,lr])

    learn.freeze_to(-2)

    lrf=learn.lr_find(lrs/1000)
    plt.title('lrs = np.array([lr/1000,lr/100,lr=2e-2]) lr_find(lrs/1000)')
    learn.sched.plot(1)
    #this plot is pretty variable, can be very different character on each run
    plt.show()

    vals = learn.fit(lrs/5, 1, cycle_len=1)
    logger.debug(vals)
    learn.unfreeze()

    vals = learn.fit(lrs/5, 1, cycle_len=2)
    logger.debug(vals)
    learn.save('clas_one')

def predict_data(learn, md):
    logger.debug('>>predict_data')
    learn.load('clas_one')

    x,y = next(iter(md.val_dl))
    probs = F.softmax(predict_batch(learn.model, x), -1)
    x,preds = to_np(x),to_np(probs)
    preds = np.argmax(preds, -1)
    return preds, x


def plot_some_images(md, preds, x):

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i,ax in enumerate(axes.flat):
        ima=md.val_ds.denorm(x)[i]
        b = md.classes[preds[i]]
        ax = show_img(ima, ax=ax)
        draw_text(ax, (0,0), b)
    plt.tight_layout()
    plt.show()

def bbox_only(trn_lrg_anno):
    #saves filename, bb coords to csv
    bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
    bbs = [' '.join(str(p) for p in o) for o in bb]

    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn','bbox'])
    #filename, 4 bb coords
    df.to_csv(BB_CSV, index=False)


def create_resnet():
    logger.debug('>>create_resnet()')
    f_model=resnet34
    sz=224
    bs=64

    tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)

    md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True)#, num_workers=0)

    # can no grab 1 bb
    x,y=next(iter(md.val_dl))

    ima=md.val_ds.denorm(to_np(x))[0]
    b = bb_hw(to_np(y[0]))
    logger.debug(f'b back to h x w for y[0]: {b}')
    ax = show_img(ima)
    draw_rect(ax, b)
    draw_text(ax, b[:2], 'label')
    plt.show()
    return f_model, md

def create_single_layer_4_out(f_model, md):
    #creating single linear with 4 outputs - tiny model, flattened
    head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088,4))
    #custom_head
    learn = ConvLearner.pretrained(f_model, md, custom_head=head_reg4)
    learn.opt_fn = optim.Adam
    #not using MSE, L1 - add up absolute vals, MSE penalises bad misses too much
    learn.crit = nn.L1Loss()

    learn.lr_find(1e-5,100)
    learn.sched.plot(5)
    plt.show()

def fit_resnet(learn):
    logger.debug('>>fit_resnet()')
    lr = 2e-3

    learn.fit(lr, 2, cycle_len=1, cycle_mult=2)

    lrs = np.array([lr/100,lr/10,lr])

    learn.freeze_to(-2)

    lrf=learn.lr_find(lrs/1000)
    learn.sched.plot(1)

    learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)

    learn.freeze_to(-3)

    learn.fit(lrs, 1, cycle_len=2)

    learn.save('reg4')

def show_preds(learn, md):
    logger.debug('>>show_preds()')
    learn.load('reg4')

    x,y = next(iter(md.val_dl))
    learn.model.eval()
    preds = to_np(learn.model(VV(x)))
    logger.debug(f'preds: {preds}')

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i,ax in enumerate(axes.flat):
        ima=md.val_ds.denorm(to_np(x))[i]
        b = bb_hw(preds[i])
        ax = show_img(ima, ax=ax)
        draw_rect(ax, b)

    #note how the box locations look crappy with more than 1 object
    plt.show()
    plt.tight_layout()
    return preds


class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): self.ds, self.y2 = ds, y2

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return (x, (y, self.y2[i]))

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

def single_object_detection():
    logger.debug('>>single_object_detection')
    # ## Single object detection
    f_model=resnet34
    sz=224
    bs=64

    val_idxs = get_cv_idxs(len(trn_fns))

    tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)
    md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
        continuous=True, num_workers=4, val_idxs=val_idxs)

    md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))

    trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
    val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)
    md.trn_dl.dataset = trn_ds2
    md.val_dl.dataset = val_ds2

    x,y=next(iter(md.val_dl))

    ima=md.val_ds.ds.denorm(to_np(x))[1]
    b = bb_hw(to_np(y[0][1]))
    logger.debug(b)

    ax = show_img(ima)
    draw_rect(ax, b)
    draw_text(ax, b[:2], md2.classes[y[1][1]])
    plt.show()
    return f_model, md, md2

def build_convnet(f_model, md):
    logger.debug(f'>>build_convnet f_model: {f_model}')
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

    learn.crit = detn_loss
    learn.metrics = [detn_acc, detn_l1]

    learn.lr_find()
    learn.sched.plot()
    plt.show()


def learn_fit(learn):
    learn_fit('>>learn_fit()')
    lr=1e-2

    learn.fit(lr, 1, cycle_len=3, use_clr=(32,5))

    learn.save('reg1_0')

    learn.freeze_to(-2)

    lrs = np.array([lr/100, lr/10, lr])

    learn.lr_find(lrs/1000)
    learn.sched.plot(0)

    learn.fit(lrs/5, 1, cycle_len=5, use_clr=(32,10))

    learn.save('reg1_1')

    learn.load('reg1_1')

    learn.unfreeze()

    learn.fit(lrs/10, 1, cycle_len=10, use_clr=(32,10))

    learn.save('reg1')
    return learn

def final_predict(learn, md, md2):
    learn_fit('>>final_predict()')
    learn.load('reg1')

    y = learn.predict()
    x,_ = next(iter(md.val_dl))

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ima = md.val_ds.ds.denorm(to_np(x))[i]
        bb = expit(y[i][:4]) * 224
        b = bb_hw(bb)
        c = np.argmax(y[i][4:])
        ax = show_img(ima, ax=ax)
        draw_rect(ax, b)
        draw_text(ax, b[:2], md2.classes[c])

    plt.tight_layout()
    plt.show()


def workflow():
    start = time.time()
    # optional 3 functions
    #im_0a = examples()
    #show_ex(im_0a)
    #draw_ex_n(17)

    trn_lrg_anno = get_trn_lrg_anno()
    #optional
    #draw_anno_n(trn_lrg_anno, 23)
    #save_annotations(trn_lrg_anno)


    #now for the interesting stuff
    f_model, md = create_model()
    learn = find_model_lr(f_model, md)
    train_model(learn)
    preds, x = predict_data(learn, md)
    plot_some_images(md, preds, x)
    bbox_only(trn_lrg_anno)
    f_model, md = create_resnet()
    create_single_layer_4_out(f_model, md)
    fit_resnet(learn)
    show_preds(learn, md)

    #second part:
    f_model, md, md2 = single_object_detection()
    build_convnet(f_model, md)
    learn = learn_fit(learn)
    #something wrong with final prediction model, needs debugging
    final_predict(learn, md, md2)
    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()