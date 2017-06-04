# ML 2017 hw6
# Plot Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model

MODEL = sys.argv[1]

def main():
    
    model = load_model(MODEL)


    if '-h' in sys.argv:

        HIS_FILE = MODEL[:-3] + '_his.npz'
        history = np.load(HIS_FILE)
        rmse = history['rmse']
        val_rmse = history['val_rmse']
        
        plt.clf()
        plt.plot(rmse, 'b')
        plt.plot(val_rmse, 'r')

        plt.legend(['RMSE', 'val RMSE'], loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('RMSE')
        plt.title(MODEL)
        plt.savefig(MODEL[:-3] + '_his.png', dpi=300)
        plt.show()

if __name__ == '__main__':
    main()
