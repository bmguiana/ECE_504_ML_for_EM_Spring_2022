"""
@author: Brian Guiana

Code snippets from the chapter 3 and chapter 4 sections of [5] were used for
the train_and_test function. The class [4] was used frequently in all aspects.

[4] ECE 504: Machine Learning for Electromagnetics.
  University Course, Spring 2022, University of Idaho, Moscow, ID.

[5] Aurelien Geron, et al, handson-ml2 [Online] github.com/ageron/handson-ml2/.
  10 Feb. 2022.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =============================================================================
# User Inputs
# =============================================================================

SNR = 0.01                   # Signal to noise ratio (V/m // V/m)
NF = 4                       # Noise figure, choose from below
                             #     0: No noise (Control)
                             #     1: Uniform noise
                             #     2: exponential noise
                             #     3: gaussian noise
                             #     4: correlated noise

# =============================================================================
# Functions
# =============================================================================

def train_and_test(X, y):
    samples = X.shape[0]
    ntrain = int(0.8*samples)

    X_train = X[:ntrain]
    X_test = X[ntrain:]
    y_train = y[:ntrain]
    y_test = y[ntrain:]

    param_grid = [{'weights': ["distance"], 'n_neighbors': [4]}]

    print('\n\ntraining')

    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    grid_search.best_score_

    print('testing\n')

    y_pred = grid_search.predict(X_test)
    final_score = accuracy_score(y_test, y_pred)

    return final_score

# =============================================================================
# Main program
# =============================================================================

print('Attempting to load data...')
if NF == 0:
    try:
        modal_data = np.load('./results/control_{}.npy'.format(SNR), allow_pickle=True).item()
        print('control data loaded')
    except:
        raise Exception('Control data unavailable. Try running gen_data_final.py with NF=0')
if NF == 1:
    try:
        modal_data = np.load('./results/uniform_{}.npy'.format(SNR), allow_pickle=True).item()
        print('uniform noise data loaded')
    except:
        raise Exception('Uniform noise data unavailable. Try running gen_data_final.py with NF=1')
if NF == 2:
    try:
        modal_data = np.load('./results/exponential_{}.npy'.format(SNR), allow_pickle=True).item()
        print('Exponential noise data loaded')
    except:
        raise Exception('Exponential noise data unavailable. Try running gen_data_final.py with NF=2')
if NF == 3:
    try:
        modal_data = np.load('./results/normal_{}.npy'.format(SNR), allow_pickle=True).item()
        print('Normal noise data loaded')
    except:
        raise Exception('Normal noise data unavailable. Try running gen_data_final.py with NF=3')
if NF == 4:
    try:
        modal_data = np.load('./results/correlated_{}.npy'.format(SNR), allow_pickle=True).item()
        print('Correltated noise data loaded')
    except:
        raise Exception('Correlated noise data unavailable. Try running gen_data_final.py with NF=4')


mag_Ex = modal_data['mag_Ex']
ph_Ex = modal_data['ph_Ex']
mag_Ey = modal_data['mag_Ey']
ph_Ey = modal_data['ph_Ey']
ms = modal_data['m']
ns = modal_data['n']
TE_TM = modal_data['mode']

del modal_data

DATA = np.hstack([mag_Ex, ph_Ex])
DATA = np.hstack([DATA, mag_Ey])
DATA = np.hstack([DATA, ph_Ey])

mag_ph_bcs = train_and_test(DATA, (TE_TM == 'E'))
mag_ph_nbcs = train_and_test(DATA, 10*ms+ns)

print('\n\nBinary Classifier Scores:')
print('Accuracy Score: ', mag_ph_bcs)

print('Multiclass Classifier Scores:')
print('Accuracy Score: ', mag_ph_nbcs)
