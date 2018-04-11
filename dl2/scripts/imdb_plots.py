import matplotlib.pylab as plt

from fastai.text import *


def load_obj(name ):
    with open('img/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot_multi_lines(x_list, y_list, n_runs, name):
    for x, y, in zip(x_list, y_list):
        plt.plot(x, y)
    plt.savefig(name)
    plt.close()

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
