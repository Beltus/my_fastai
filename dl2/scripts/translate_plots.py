from timeit import default_timer as timer

import matplotlib.pylab as plt
from matplotlib.pyplot import cm

from fastai.text import *

from .plot_functions import plot_all_ep_vals, plot_all_ep_train_val, \
    load_obj, scatter_plot_final_val, box_plot_final_val, plot_all_in_range



def plot_method_comparisson(arch = 's2s', rnn='GRU'):
    ep_val_dict = {}
    ep_val_dict[f'{arch}_default'] = load_obj(f'translate_ep_vals_{arch}')
    ep_val_dict[f'{arch}_tuned'] = load_obj(f'translate_ep_vals_{arch}_{rnn}_plot_params')
    plot_all_ep_vals(ep_val_dict, img_file_name='translate_{arch}_all_ep_vals_default_vs_tuned.png')

def plot_method_comparisson_0(arch = 's2s', rnn='GRU', default=True):
    ep_val_dict = {}
    ep_val_dict['no_dropout'] = load_obj(f'translate_ep_vals_{arch}_{rnn}_all_drop_0')
    if default:
        ep_val_dict['default'] = load_obj(f'translate_ep_vals_{arch}')
        file_indicator = 'default'
    else:
        ep_val_dict['tuned'] = load_obj(f'translate_ep_vals_{arch}_{rnn}_plot_params')
        file_indicator = 'tuned'
    plot_all_ep_train_val(ep_val_dict, img_file_name=f'translate_{arch}_all_ep_vals_{file_indicator}_vs_all_drop_0.png')

def plot_default_params():
    ep_val_dict = {}
    ep_val_dict['seq2seq'] = load_obj('translate_ep_vals_s2s')
    ep_val_dict['bdir'] = load_obj('translate_ep_vals_bdir')
    ep_val_dict['teacher'] = load_obj('translate_ep_vals_force')
    ep_val_dict['attn'] = load_obj('translate_ep_vals_attn')
    ep_val_dict['all'] = load_obj('translate_ep_vals_all')
    plot_all_ep_vals(ep_val_dict, img_file_name= 'translate_all_ep_vals_default.png')

def batch_val_plots(arch='attn', perplex=False):
    range_strt=1
    range_stop=10
    for acron in ['eed', 'od', 'rdd', 'red']:
        plot_all_in_range('translate', acron, range_strt, range_stop, arch, perplex)

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

def temp_batch_val_plots(arch='attn'):
    #run not finished, temp plots
    #TODO set range stop to 10
    range_strt=1
    range_stop=10
    for acron in ['eed', 'od', 'rdd', 'red']:
        plot_all_in_range(acron, range_strt, range_stop, arch)

def workflow():
    start = timer()
    #plot_default_params()
    #TODO, find best params then create a trained
    #plot_method_comparisson(arch = 's2s', rnn='GRU')

    #compare to no dropout, already gen (TODO s2s tuned)
    #plot_method_comparisson_0(arch = 's2s', rnn='GRU', default=True)
    #plot_method_comparisson_0(arch='attn', rnn='GRU', default=True)
    #plot_method_comparisson_0(arch='attn', rnn='GRU', default=False)

    #have already generated these
    #batch_val_plots('attn')
    #temp_batch_val_plots(arch = 's2s_GRU')

    #run 1-10 and 1-6
    #batch_val_plots(arch='attn_GRU_all_drop_0', perplex=False)
    batch_val_plots(arch = 's2s_GRU_all_drop_0')

    #batch_final_val_plots('translate_ep_vals_attn', arch_type='attn')


    #batch_seq2seq_val_plots()
    #batch_final_val_plots('translate_ep_vals_s2s_GRU', arch_type='s2s')

    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()
