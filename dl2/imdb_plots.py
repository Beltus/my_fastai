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

file_prefix = 'data_subset_sens_full_3_epoch_'
files = ['5', '10', '50']

data_dict = {}
for i in files:
    file_name = file_prefix+str(i)
    data = load_obj(file_name)
    data_dict[i] = data

print (data_dict)