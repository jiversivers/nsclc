# Import packages
import os
import random

import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
from sklearn.metrics import auc, ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score, roc_curve

from my_modules.custom_models import *
from my_modules.nsclc import subdivide_list
from my_modules.nsclc.nsclc_dataset import NSCLCDataset


def main():
    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare data
    data = NSCLCDataset('data/NSCLC_Data_for_ML',
                        ['orr', 'g', 's', 'photons', 'tau1', 'tau2', 'alpha1', 'alpha2', 'taumean', 'boundfraction'],
                        device='cpu', label='Metastases', mask_on=False)
    data.normalize_channels('preset')
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.to(torch.device(device))
    data.augment()

    # Dataloader parameters
    batch_size = 32

    # Set up hyperparameters
    epochs = 1000
    lr = 1e-6

    # Set up training functions
    optim_fn = optim.Adam
    loss_function = nn.BCEWithLogitsLoss()

    # Model zoo for images
    model = RegularizedParallelCNNet

    # Prepare folded data samplers
    # Get random indices of patients
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(data.patient_count))
    idx = [i for i in subsampler]

    # Get the image indices for all patients as nested lists
    patient_subsets = [data.get_patient_subset(i) for i in idx]

    # Find and remove any patients with no image indices
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
    for ix in idx_for_removal:
        idx.remove(ix)

    # Get labels for all remaining patients
    labels = [data.get_patient_label(i).item() for i in idx]

    # Separate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]
    print('Number of non-metastatic patients: {}'.format(len(shuffled_ones)))
    print('Number of metastatic patients: {}'.format(len(shuffled_zeros)))

    # Determine number of folds such that at least 3 unique patients from each class can be present in all folds
    dividend = len(shuffled_zeros) if len(shuffled_zeros) < len(shuffled_ones) else len(shuffled_ones)
    num_folds = 1
    while dividend / num_folds > 3:
        num_folds += 1
    print('Number of data folds to ensure at least n=3 per class for most classes: {}'.format(num_folds))

    # Fold data
    folded_ones = subdivide_list(shuffled_ones, num_folds)
    folded_zeros = subdivide_list(shuffled_zeros, num_folds)
    print('Number of non-metastatic patients per fold in test set: {}'.format([len(fo) for fo in folded_ones]))
    print('Number of metastatic patients per fold in test set: {}'.format([len(fz) for fz in folded_zeros]))

    train_folds = []
    test_folds = []
    for fold in range(4):
        # Get subsets for this fold
        train_subjects = []
        for sub in range(num_folds):
            if sub != fold:
                train_subjects = train_subjects + folded_zeros[sub] + folded_ones[sub]  # Get all that aren't test-set
        train_subsets = [data.get_patient_subset(i) for i in train_subjects]  # Get all patient indices
        train_indices = [i for sub in train_subsets for i in sub]  # Un-nest
        random.shuffle(train_indices)

        test_subjects = folded_zeros[fold] + folded_ones[fold]  # Get a set of patients from both classes
        test_subsets = [data.get_patient_subset(i) for i in test_subjects]  # Get all patient indices
        test_indices = [i for sub in test_subsets for i in sub]  # Un-nest
        random.shuffle(test_indices)

        # Make data subsets from indices-subsets
        train_folds.append(torch.utils.data.Subset(data, train_indices))
        test_folds.append(torch.utils.data.Subset(data, test_indices))

    for fold, (train, test) in enumerate(zip(train_folds, test_folds)):
        print(f'Final dataset split for fold {fold + 1} - Training set: {len(train)}, Test set: {len(test)}'
              f' - {len(train) / len(data):0.2f} : {len(test) / len(data):0.2f}')

    # endregion

    # Prep results file
    results_file_path = 'results.txt'
    with open(results_file_path, 'a') as results_file:
        results_file.write('\nAugmented Images\n')
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    # Model path
    try:
        os.mkdir('aug_img_models')
    except FileExistsError:
        pass

    training_loss = []
    aucs = []
    bacs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold in range(num_folds):
        m = model(data.shape)

        # Create dataloaders for fold
        train_loader = torch.utils.data.DataLoader(train_folds[fold],
                                                   batch_size=batch_size, shuffle=True, num_workers=0,
                                                   drop_last=(True if len(train_folds) % batch_size == 1 else False))
        test_loader = torch.utils.data.DataLoader(test_folds[fold],
                                                  batch_size=batch_size, shuffle=False, num_workers=0,
                                                  drop_last=(True if len(test_folds) % batch_size == 1 else False))

        if torch.cuda.is_available() and not next(m.parameters()).is_cuda:
            m.to(torch.device(device))

        optimizer = optim_fn(m.parameters(), lr=lr)
        print(f'Training fold {fold + 1}')
        print('_____________________________________________________________________________________________\n')
        # Train
        training_loss.append([])
        m.train()
        for ep in range(epochs):
            total_loss = 0
            for x, target in train_loader:
                optimizer.zero_grad()
                out = m(x)
                loss = loss_function(out, target.unsqueeze(1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            training_loss[-1].append(total_loss)
            with open(results_file_path, 'a') as results_file:
                results_file.write(f'\nEpoch {ep + 1}: Train.Loss: {training_loss[-1][-1] / len(train_folds[fold]):.4f}')

        torch.save(m.state_dict(), f'aug_img_models/{data.name}__{m.name}__{lr}_{ep}__fold{fold + 1}.pth')

        # Testing
        m.eval()
        outs = torch.tensor([])
        targets = torch.tensor([])
        with torch.no_grad():
            for x, target in test_loader:
                if torch.cuda.is_available() and not x.is_cuda:
                    x = x.cuda()
                    target = target.cuda()
                outs = torch.cat((outs, m(x).cpu().detach()), dim=0)
                targets = torch.cat((targets, target.cpu().detach()), dim=0)

        # AUC
        fpr, tpr, thresholds = roc_curve(targets, outs, pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))  # Interpolate tpr at common fprs
        tprs[-1][0] = 0.0  # Ensure the ROC starts at (0, 0)
        aucs.append(auc(fpr, tpr))

        # Balanced accuracy
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        preds = torch.zeros_like(outs)
        preds[outs > best_thresh] = 1
        bacs.append(balanced_accuracy_score(targets, preds))
        print(f'>>> {m.name} for fold {fold + 1} -- '
              f'AUC: {aucs[-1]:.4f} | BAC: {100 * bacs[-1]:.2f}%  at threshold: {best_thresh:.4f}.')

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucs[-1]).plot(ax=ax1)
        ax1.set_title(f'ROC for Fold {fold + 1}')
        ConfusionMatrixDisplay.from_predictions(targets, preds).plot(ax=ax2)
        ax2.set_title(f'Confusion Matrix for Fold {fold + 1}')
        fig.savefig(f'outputs/plots/fold{fold + 1}_roc_big-comet.png')
        plt.close(fig)

        # Write
        with open(results_file_path, 'a') as f:
            f.write(f'Fold {fold + 1} -- AUC: {aucs[-1]:.4f}'
                    f' | BAC: {100 * bacs[-1]:.2f}%  at threshold: {best_thresh:.4f}.\n')

    # Calc mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure end of ROC at (1, 1)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
    tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 1)

    print(f'Mean AUC -- from AUCS: {np.mean(aucs):.4f} -- from rates: {mean_auc:.4f} | Std.Dev. AUCS: {std_auc:.4f}')

    # Plot
    plt.close('all')
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=5, alpha=0.8,
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    for tpr in tprs:
        plt.plot(mean_fpr, tpr, lw=2.5, alpha=0.3)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('outputs/plots/mean_roc_big-comet.png')
    plt.close()

    # Write
    with open('outputs/results.txt', 'a') as f:
        f.write(
            f'Mean AUC -- from AUCS: {np.mean(aucs):.4f} -- from rates: {mean_auc:.4f} | Std.Dev. AUCS: {std_auc:.4f}')
# Run
if __name__ == '__main__':
    main()
