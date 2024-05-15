import torch.optim as optim

from my_modules.custom_models import *
from my_modules.model_learning import single_model_iterator
from my_modules.nsclc.nsclc_dataset import NSCLCDataset

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
    data.augment()
    data.show_random()

    # Prepare training/data-loading parameters
    optim_fun = optim.SGD
    criterion = nn.BCELoss()
    bs = 32  # Batch size
    kf = 5  # Number of folds for cross validation

    # Iterable hyperparameters
    hyperparameters = {'Augmented': {'Fast': {'LR': 0.01, 'EP': 125}},
                       'Raw': {'Fast': {'LR': 0.01, 'EP': 125}}}

    status = ['Metastases', 'Response']

    results = {'Image': {'Single': {}, 'KFold': {}},
               'Hist': {'Single': {}, 'KFold': {}}}
    results_file_path = 'results.txt'
    with open(results_file_path, 'w') as results_file:
        results_file.write('Results')

    # Iterate image-based classifiers (CNNs, no RNNs)
    data.dist_transformed = False  # This will revert back to returning images
    models = [ParallelCNNet, RegularizedParallelCNNet, CNNet,
              MLPNet, RegularizedMLPNet, ParallelMLPNet, RegularizedParallelMLPNet]
    for aug, styles in hyperparameters.items():
        data.augmented = True if aug == 'Augmented' else False
        for key in status:
            data.label = key
            for style, hp in styles.items():
                print(f'Currently training {aug} data at {style} rate on {key}...')
                results['Image']['Single'][style] = single_model_iterator(models, data,
                                                                          hp['EP'], bs, criterion,
                                                                          optim_fun, lr=hp['LR'],
                                                                          num_workers=(16, 16, 8),
                                                                          pin_memory=True, momentum=0.9)
                with open(results_file_path, 'a') as f:
                    f.write(f'{aug} {style}: {results["Image"]["Single"][style]}\n')

    # Show results from image-based classification
    for key, item in results['Image']['Single'].items():
        print(f'{key} {item}%')

    # Iterate distribution-compatible models (RNNs, no CNNs)
    data.dist_transform()
    models = [RNNet, RegularizedRNNet, ParallelMLPNet, RegularizedMLPNet, RegularizedParallelMLPNet, MLPNet]
    for aug, styles in hyperparameters.items():
        data.augmented = True if aug == 'Augmented' else False
        for key in status:
            data.label = key
            for style, hp in styles.items():
                print(f'Currently training {aug} data at {style} rate on {key}...')
                results['Hist']['Single'][style] = single_model_iterator(models, data,
                                                                         hp['EP'], bs, criterion,
                                                                         optim_fun, lr=hp['LR'],
                                                                         num_workers=(16, 16, 8))
                with open(results_file_path) as f:
                    f.write(f'{aug} {style}: {results["Image"]["Single"][style]}')

    # Show results from dist-based classification
    for key, item in results['Hist']['Single'].items():
        print(f'{key} {item}%')


if __name__ == "__main__":
    main()
