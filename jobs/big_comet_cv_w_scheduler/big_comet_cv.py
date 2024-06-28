import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms.v2 as tvt
from pretrainedmodels import inceptionresnetv2
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, RocCurveDisplay, ConfusionMatrixDisplay

from my_modules.custom_models import CometClassifierWithBinaryOutput, FeatureExtractorToClassifier as FETC
from my_modules.nsclc import NSCLCDataset, subdivide_list


def main():
    #region Prepare environment
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    with open('outputs/results.txt', 'w') as results_file:
        results_file.write('Cross Validation Results\n')
    #endregion

    #region Set up the dataset
    # Images, no mask (feature extractor will hopefully handle this), normalized_to_max (already is),
    data = NSCLCDataset('NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=True)
    data.augment()
    data.normalize_channels('preset')
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.to(device)

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
    print('Number of data folds to ensure at least n=3 per class: {}'.format(num_folds))

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
              f' - {len(train) / len(data):0.2f} : {len(test) / len(data):0.2f} ')

    #endregion

    #region Prepare model
    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = inceptionresnetv2(num_classes=1001, pretrained=False)

    # Load pretrained from download
    feature_extractor.load_state_dict(
        torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/inceptionresnetv2-520b38e4.pth'))
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Define base classifier
    classifier = CometClassifierWithBinaryOutput
    #endregion

    # Hyerparameters
    batch_size = 64
    lr = 0.01
    optimizer_fn = torch.optim.RMSprop
    epochs = 1000
    loss_fn = torch.nn.BCELoss()

    training_loss = []
    aucs = []
    bacs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold in range(num_folds):
        # Create dataloaders for fold
        train_loader = torch.utils.data.DataLoader(train_folds[fold],
                                                   batch_size=batch_size, shuffle=True, num_workers=0,
                                                   drop_last=(True if len(train_folds) % batch_size == 1 else False))
        test_loader = torch.utils.data.DataLoader(test_folds[fold],
                                                  batch_size=batch_size, shuffle=False, num_workers=0,
                                                  drop_last=(True if len(test_folds) % batch_size == 1 else False))

        # Init full model
        model = FETC(data.shape, feature_extractor=feature_extractor, classifier=classifier,
                     layer='conv2d_7b')
        model.to(device)

        # Make optimizer at the current larning rate with only classifier parameters
        optimizer = optimizer_fn(model.classifier.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               cooldown=5, min_lr=5e-6)
        current_lr = lr

        # Training
        # Nest iteration lists for tracking model learning
        training_loss.append([])
        for ep in range(epochs):
            # Training
            epoch_loss = 0
            model.train()
            for x, target in train_loader:
                if torch.cuda.is_available() and not x.is_cuda:
                    x = x.cuda()
                if torch.cuda.is_available() and not target.is_cuda:
                    target = target.cuda()

                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, target.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            training_loss[-1].append(epoch_loss / len(train_folds[fold]))
            scheduler.step(training_loss[-1][-1])
            if scheduler.get_last_lr() != current_lr:
                current_lr = scheduler.get_lr()
                print(f'Updated LR at {ep + 1} to {current_lr}')

            with open('outputs/results.txt', 'a') as results_file:
                results_file.write(f'\nEpoch {ep + 1}: Train.Loss: {training_loss[-1][-1]:.4f}, ')

        torch.save(model.state_dict(), f'outputs/models/{data.name}__{model.name}__{lr}_{ep}__Fold{fold + 1}.pth')

        # Testing
        model.eval()
        outs = torch.tensor([])
        targets = torch.tensor([])
        with torch.no_grad():
            for x, target in test_loader:
                if torch.cuda.is_available() and not x.is_cuda:
                    x = x.cuda()
                    target = target.cuda()
                outs = torch.cat((outs, model(x).cpu().detach()), dim=0)
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
        print(f'>>> {model.name} for fold {fold + 1} -- '
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
        with open('outputs/results.txt', 'a') as f:
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

if __name__ == '__main__':
    main()
