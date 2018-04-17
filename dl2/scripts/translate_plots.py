from timeit import default_timer as timer

import matplotlib.pylab as plt
from matplotlib.pyplot import cm

from fastai.text import *


def save_obj(obj, name ):
    with open('../data_tests/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data_tests/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def box_plot_final_val(title, final_val_dict, img_file_name):
    label_list = []
    val_list = []
    for k, semi_final_val_dict in final_val_dict.items():
        keys = semi_final_val_dict.keys()
        vals = semi_final_val_dict.values()
        k_lbl = []
        for ke in keys:
            print(ke)
            k_lbl.append(int(ke[-1:])/10)
        label_list.append(k)
        val_list.append(list(vals))

    print(f'label_list: {label_list}, val_list: {val_list}')
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    # rectangular box plot
    bplot = plt.boxplot(val_list,
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=label_list)  # will be used to label x-ticks
    plt.title('Dropout type impact')
    plt.ylabel('Validation loss')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'lightgrey']
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    plt.savefig('../img/' + img_file_name)
    plt.close()

def scatter_plot_final_val(title, final_val_dict, img_file_name):
    color = iter(cm.rainbow(np.linspace(0, 1, len(final_val_dict))))
    for k, semi_final_val_dict in final_val_dict.items():
        keys = semi_final_val_dict.keys()
        vals = semi_final_val_dict.values()
        k_lbl = []
        for ke in keys:
            print(ke)
            k_lbl.append(int(ke[-1:])/10)
        c = next(color)
        plt.plot(k_lbl, vals, c=c, label=k)
    plt.ylabel("val_loss")
    plt.xlabel("dropout scalar")
    plt.legend(loc='upper left')
    plt.savefig('../img/' + img_file_name)
    plt.close()

def plot_all_ep_vals(ep_val_dict, img_file_name):
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
    plt.savefig('../img/'+img_file_name)
    plt.close()

def plot_method_comparisson():
    ep_val_dict = {}
    ep_val_dict['attn_default'] = load_obj('translate_ep_vals_attn')
    ep_val_dict['attn_tuned'] = load_obj('translate_ep_vals_attn_GRU_plot_params')
    plot_all_ep_vals(ep_val_dict, img_file_name='translate_all_ep_vals_default_vs_tuned.png')

def plot_default_params():
    ep_val_dict = {}
    ep_val_dict['seq2seq'] = load_obj('translate_ep_vals_s2s')
    ep_val_dict['bdir'] = load_obj('translate_ep_vals_bdir')
    ep_val_dict['teacher'] = load_obj('translate_ep_vals_force')
    ep_val_dict['attn'] = load_obj('translate_ep_vals_attn')
    ep_val_dict['all'] = load_obj('translate_ep_vals_all')
    plot_all_ep_vals(ep_val_dict, img_file_name= 'translate_all_ep_vals_default.png')

def plot_attn(drop_acron, range_strt, range_stop):
    ep_val_dict = {}
    for i in range(range_strt, range_stop):
        pt_txt = str(i/10)
        ep_val_dict[f'{drop_acron}_{pt_txt}'] = load_obj(f'translate_ep_vals_attn_{drop_acron}_{i}')
    plot_all_ep_vals(ep_val_dict, img_file_name=f'translate_attn_{drop_acron}.png')

def batch_attn_val_plots():
    range_strt=1
    range_stop=6
    for acron in ['eed', 'od', 'rdd', 'red']:
        plot_attn(acron, range_strt, range_stop)

def batch_final_val_plots(file_base, arch_type):
    range_strt=1
    range_stop=10
    final_val_dict = {}
    for drop_acron in ['eed', 'od', 'rdd', 'red']:
        semi_final_val_dict = {}
        ep_val_dict = {}
        ep_data_dict = {}
        pt_txt = 'final_at_'+str(range_stop)
        for i in range(range_strt, range_stop):
            ep_val_dict[f'{drop_acron}_{i}'] = load_obj(f'{file_base}_{drop_acron}_{i}')
        for k, v in ep_val_dict.items():
            sub_dict = v
            key_lst = list(sub_dict.keys())
            val_lst = list(sub_dict.values())
            final_val = val_lst[-1]
            semi_final_val_dict[k] = final_val[1]
        final_val_dict[drop_acron] = semi_final_val_dict
    scatter_plot_final_val(drop_acron, final_val_dict, img_file_name=f'translate_{arch_type}_final_{range_strt}-{range_stop}')
    box_plot_final_val(drop_acron, final_val_dict, img_file_name=f'translate_{arch_type}_final_{range_strt}-{range_stop}_box')

def plot_seq2seq(drop_acron, range_strt, range_stop, postfix):
    ep_val_dict = {}
    for i in range(range_strt, range_stop):
        pt_txt = str(i/10)
        ep_val_dict[f'{drop_acron}_{pt_txt}'] = load_obj(f'translate_ep_vals_s2s_GRU_{drop_acron}_{i}')
    plot_all_ep_vals(ep_val_dict, img_file_name=f'translate_s2s_GRU_{drop_acron}_{postfix}.png')

def batch_seq2seq_val_plots():
    range_strt=1
    range_stop=10
    postfix = str(range_strt)+'_'+str(range_stop)
    for acron in ['eed', 'od', 'rdd', 'red']:
        plot_seq2seq(acron, range_strt, range_stop, postfix)

def workflow():
    start = timer()
    #plot_default_params()
    #plot_method_comparisson()
    #batch_attn_val_plots()
    #batch_final_val_plots('translate_ep_vals_attn', arch_type='attn')
    #batch_seq2seq_val_plots()
    batch_final_val_plots('translate_ep_vals_s2s_GRU', arch_type='s2s')
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()
