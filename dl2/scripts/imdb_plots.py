from timeit import default_timer as timer

import matplotlib.pylab as plt

from fastai.text import *

from plot_functions import plot_all_ep_vals, plot_all_ep_train_val, \
    load_obj, scatter_plot_final_val, box_plot_final_val, plot_all_in_range, plot_all_in_range_decimal


def load_obj(name ):
    with open('../data_tests/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot_multi_lines(x_list, y_list, n_runs, name):
    for x, y, in zip(x_list, y_list):
        plt.plot(x, y)
    plt.savefig(name)
    plt.close()

def data_subset_sens_full_3():
    data_dict = {}

    file_prefix = 'data_subset_sens_full_3_no_pt_epoch_'
    files = ['1', '2', '10', '25', '50', '100', '200']

    for i in files:
        file_name = file_prefix+str(i)
        data = load_obj(file_name)
        data_dict[i] = data

    #data = load_obj('data_subset_sens_full_3_epoch')
    #data_dict[i] = data
    print (data_dict)

def batch_val_plots(arch='attn', perplex=False):
    range_strt = 5
    range_stop = 30
    for acron in ['di']:
        plot_all_in_range_decimal(arch, acron, range_strt, range_stop, arch, perplex)

def batch_final_val_plots(file_base, arch_type):
    range_strt=5
    range_stop=30
    final_val_dict = {}
    for drop_acron in ['di']:
        semi_final_val_dict = {}
        ep_val_dict = {}
        ep_data_dict = {}
        pt_txt = 'final_at_'+str(range_stop)
        for i in range(range_strt, range_stop, 5):
            num_str = str(i/100)
            print(num_str)
            ep_val_dict[f'{drop_acron}_{i}'] = load_obj(f'{file_base}_{drop_acron}_{num_str}')
        for k, v in ep_val_dict.items():
            print(f'k: {k}, v: {v}')
            sub_dict = v
            key_lst = list(sub_dict.keys())
            val_lst = list(sub_dict.values())
            final_val = val_lst[-1]
            semi_final_val_dict[k] = final_val[1]
        final_val_dict[drop_acron] = semi_final_val_dict
    print(final_val_dict)
    scatter_plot_final_val(drop_acron, final_val_dict, img_file_name=f'imdb_{arch_type}_{range_strt}-{range_stop}')
    box_plot_final_val(drop_acron, final_val_dict, img_file_name=f'imdb_{arch_type}_final_{range_strt}-{range_stop}_box')


def workflow():
    start = timer()
    #batch_final_val_plots('imdb_ep_vals_full_di_d_de_dh_drop_0_dw_0pt001', 'full')
    batch_val_plots('imdb_ep_vals_full_di_d_de_dh_drop_0_dw_0pt001')
    end = timer()
    elapsed = end - start
    print(f'>>workflow() took {elapsed}sec')


if __name__ == "__main__":
    workflow()