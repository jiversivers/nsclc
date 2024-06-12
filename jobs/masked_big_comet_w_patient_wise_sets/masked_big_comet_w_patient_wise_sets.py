import os

import torch
import random
from pretrainedmodels import inceptionresnetv2
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import CometClassifierWithBinaryOutput, FeatureExtractorToClassifier as FETC



def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    with open('outputs/results.txt', 'w') as results_file:
        results_file.write('Results')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = inceptionresnetv2(num_classes=1001, pretrained=False)

    # Load pretrained from download
    feature_extractor.load_state_dict(
        torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/inceptionresnetv2-520b38e4.pth'))
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Set up the dataset for this model
    # Images, no mask (feature extractor will hopefully handle this), normalized_to_max (already is),
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=True)
    print('Normalizing data to channel max...')
    data.augment()
    data.normalize_channels('max')
    data.to(device)

    batch_size = 64
    learning_rates = [5e-5, 1e-5, 5e-6, 1e-6]
    optimizer_fns = {'Adam': [torch.optim.Adam, {}]}
    epochs = [5, 10, 50, 125, 250, 500, 1000, 2500]
    loss_fn = torch.nn.BCELoss()

    # Define base classifier
    classifier = CometClassifierWithBinaryOutput

    # Split data by patients, ensuring 3 patients per class in test set
    # Sample patients randomly
    subsampler = torch.utils.data.sampler.SubsetRandomSampler(range(data.patient_count))
    idx = [i for i in subsampler]

    # Get the image indices for all patients as nested lists
    patient_subsets = [data.get_patient_subset(i) for i in idx]

    # Finda remove any patients with no image indices
    idx_for_removal = []
    for i, subset in enumerate(patient_subsets):
        if len(subset) == 0:
            idx_for_removal.append(idx[i])
    for ix in idx_for_removal:
        idx.remove(ix)

    # Get labels for all remaining patients
    labels = [data.get_patient_label(i).item() for i in idx]

    # Seperate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]

    # Assign first three of each class to test set and expand to full image indices
    test_subjects = shuffled_zeros[:3] + shuffled_ones[:3]
    test_subs = [data.get_patient_subset(i) for i in test_subjects]
    test_indices = [i for sub in test_subs for i in sub]

    # Assign remaining patients to train set
    train_subjects = shuffled_zeros[3:] + shuffled_ones[3:]
    train_subs = [data.get_patient_subset(k) for k in train_subjects]
    train_indices = [i for sub in train_subs for i in sub]

    # Shuffle and subset
    random.shuffle(test_indices)
    random.shuffle(train_indices)
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    training_loss = []
    evaluation_loss = []
    evaluation_accuracy = []
    testing_accuracy = []
    for lr in learning_rates:
        # Make full model
        model = FETC(data.shape, feature_extractor=feature_extractor, classifier=classifier,
                     layer='conv2d_7b')
        model.to(device)

        # Make optimizer at the current larning rate with only classifier parameters
        optimizer = optimizer_fns['Adam'][0](model.classifier.parameters(), lr=lr)

        # Nest iteration lists for tracking model learning
        training_loss.append([])
        evaluation_accuracy.append([])
        evaluation_loss.append([])
        testing_accuracy.append([])

        # Find max number of epochs to consider and do that, checking along the way for others
        epoch = sorted(epochs)[-1]
        for ep in range(epoch):
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
            training_loss[-1].append(epoch_loss / len(train_set))

            # See if we are at one of our training length checkpoints. Save and test if we are
            if ep + 1 in epochs:
                torch.save(model.state_dict(), f'outputs/models/{data.name}__{model.name}__{lr}_{ep + 1}.pth')
                print(f'>>> {model.name} for {ep + 1} epochs with learning rate of {lr}...')
                scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
                fig.savefig(f'outputs/plots/auc_acc_big-comet_{ep + 1}-Epochs_{lr}-LearningRate.png')
                plt.close(fig)
                with open('outputs/results.txt', 'a') as f:
                    f.write(f'\n>>> {model.name} for {ep + 1} epochs with learning rate of {lr}\n')
                    for key, item in scores.items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')

if __name__ == '__main__':
    main()
