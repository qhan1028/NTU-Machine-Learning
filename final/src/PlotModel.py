# ML2017 final
# Plot Model

import numpy as np
import matplotlib.pyplot as plt


HIS_DIR = '../history'

def plot_gt(sj_pred, sj_gt, iq_pred, iq_gt, val_sj, val_iq):
    
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(hspace=0.4)
    
    plt.subplot(2, 1, 1)
    plt.plot(sj_gt, 'b')
    plt.plot(sj_pred, 'r')
    plt.grid(linestyle=':')
    plt.ylabel('total cases')
    plt.legend(['ground truth', 'prediction'], loc='upper left', fontsize=8)
    plt.title('sj')

    plt.subplot(2, 1, 2)
    plt.plot(iq_gt, 'b')
    plt.plot(iq_pred, 'r')
    plt.grid(linestyle=':')
    plt.ylabel('total cases')
    plt.legend(['ground truth', 'prediction'], loc='upper left', fontsize=8)
    plt.title('iq')

    plt.savefig(HIS_DIR + '/sj_' + val_sj + '_iq_' + val_iq + '_gt.png', dpi=300)


def plot_history(his_sj, his_iq, val_sj, val_iq):

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.plot(his_sj['mean_absolute_error'], 'b')
    plt.plot(his_sj['val_mean_absolute_error'], 'r')
    plt.grid(linestyle=':')
    plt.xlabel('epochs')
    plt.ylabel('mean absolute error')
    plt.legend(['training', 'validation'], loc='lower left', fontsize=8)
    plt.title('sj')

    plt.subplot(2, 1, 2)
    plt.plot(his_iq['mean_absolute_error'], 'b')
    plt.plot(his_iq['val_mean_absolute_error'], 'r')
    plt.grid(linestyle=':')
    plt.xlabel('epochs')
    plt.ylabel('mean absolute error')
    plt.legend(['training', 'validation'], loc='lower left', fontsize=8)
    plt.title('iq')

    plt.savefig(HIS_DIR + '/sj_' + val_sj + '_iq_' + val_iq + '.png', dpi=300)
