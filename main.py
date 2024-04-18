from my_modules.nsclc.nsclc_dataset import NSCLCDataset
from my_modules.custom_models import *
from my_modules.model_learning import single_model_iterator

import torch.optim as optim
from torch.nn import RNN

'''
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import shuffle
import sklearn.feature_selection
from sklearn import metrics
'''


def main():
    # Prepare data
    data = NSCLCDataset('E:\\NSCLC Data - PMD', ['orr', 'photons', 'taumean', 'boundfraction'], label='Response')
    data.normalize_channels_to_max()
    data.dist_transform()
    data.show_random()

    # Prepare training/data-loading parameters
    optim_fun = optim.Adam
    criterion = nn.BCELoss()
    bs = 32  # Batch size
    kf = 5  # Number of folds for cross validation

    # Iterable hyperparameters
    hyperparameters = {'Raw': {'Fast': {'LR': 0.01, 'EP': 125},
                               'Mid': {'LR': 0.001, 'EP': 500},
                               'Slow': {'LR': 0.00001, 'EP': 2500}},
                       'Augmented': {'Fast': {'LR': 0.01, 'EP': 125},
                                     'Mid': {'LR': 0.001, 'EP': 500},
                                     'Slow': {'LR': 0.00001, 'EP': 2500}}}

    status = ['Response', 'Metastases']

    RESULTS = {'Image': {'Single': {}, 'KFold': {}},
               'Hist': {'Single': {}, 'KFold': {}}}

    # Iterate distribution-compatible models (RNNs, no CNNs)
    models = [ParallelMLPNet, RegularizedMLPNet, RegularizedParallelMLPNet, RNN, RegularizedRNNet, MLPNet]
    for aug, styles in hyperparameters.items():
        data.augmented = True if aug == 'Augmented' else False
        for key in status:
            data.label = key
            for style, hp in styles.items():
                print(f'Currently training {aug} data at {style} rate on {key}...')
                RESULTS['Hist']['Single'][style] = single_model_iterator(models, {key: data},
                                                                         hp['EP'], bs, criterion,
                                                                         optim_fun, lr=hp['LR'], num_workers=(8, 8, 8))

    # Show results from dist-based classification
    for key, item in RESULTS['Hist']['Single'].items():
        print(f'{key} {item}%')

    # Iterate image-based classifiers (CNNs, no RNNs)
    data.dist_transformed = False  # This will revert back to returning images
    data.show_random()
    models = [CNNet, RegularizedCNNet, ParallelCNNet, RegularizedParallelCNNet,
              MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet]
    for aug, styles in hyperparameters.items():
        data.augmented = True if aug == 'Augmented' else False
        for key in status:
            data.label = key
            for style, hp in styles.items():
                print(f'Currently training {aug} data at {style} rate on {key}...')
                RESULTS['Image']['Single'][style] = single_model_iterator(models, {key: data},
                                                                          hp['EP'], bs, criterion,
                                                                          optim_fun, lr=hp['LR'], num_workers=(8, 8, 8))

    # Show results from image-based classification
    for key, item in RESULTS['Image']['Single'].items():
        print(f'{key} {item}%')


if __name__ == "__main__":
    main()
