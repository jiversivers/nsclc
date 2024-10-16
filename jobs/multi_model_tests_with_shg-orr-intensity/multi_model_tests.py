# Import packages
import os

import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms.v2 as tvt
from matplotlib import pyplot as plt
import random

from my_modules.custom_models import *
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import set_seed
from my_modules.nsclc.nsclc_dataset import NSCLCDataset


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Set up multiprocessing
    print(f'Num cores: {mp.cpu_count()}')
    print(f'Num GPUs: {torch.cuda.device_count()}')
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################
    # Prepare data #
    ################
    data = NSCLCDataset('NSCLC_Data_for_ML', ['shg', 'orr', 'intensity'], device=torch.device('cpu'),
                        label='Metastases', mask_on=True)
    data.augment()
    data.normalize_method = 'preset'
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.to(device)

    # Random split datasets
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
    image_counts = [0, 0]
    for i, label in zip(idx, labels):
        image_counts[int(label)] += len(data.get_patient_subset(i))

    # Separate 0 and 1 labels (still shuffled)
    shuffled_zeros = [i for i, l in zip(idx, labels) if l == 0]
    shuffled_ones = [i for i, l in zip(idx, labels) if l == 1]
    print(f'Total non-metastatic patients: {len(shuffled_ones)} with {image_counts[1]} images')
    print(f'Total metastatic patients: {len(shuffled_zeros)} with {image_counts[0]} images')

    # Split train, eval, and test sets
    eval_pts = shuffled_zeros[0:3] + shuffled_ones[0:3]
    eval_idx = [data.get_patient_subset(i) for i in eval_pts]
    eval_idx = [im for i in eval_idx for im in i]
    random.shuffle(eval_idx)
    image_counts = [0, 0]
    for idx in eval_pts:
        label = data.get_patient_label(idx)
        image_counts[int(label)] += len(data.get_patient_subset(idx))

    train_pts = shuffled_zeros[3:-1] + shuffled_ones[3:-1]
    train_idx = [data.get_patient_subset(i) for i in train_pts]
    train_idx = [im for i in train_idx for im in i]
    random.shuffle(train_idx)
    image_counts = [0, 0]
    for idx in train_pts:
        label = data.get_patient_label(idx)
        image_counts[int(label)] += len(data.get_patient_subset(idx))

    test_pts = [shuffled_zeros[-1], shuffled_ones[-1]]
    test_idx = [data.get_patient_subset(i) for i in train_pts]
    test_idx = [im for i in test_idx for im in i]
    random.shuffle(test_idx)
    image_counts = [0, 0]
    for idx in test_pts:
        label = data.get_patient_label(idx)
        image_counts[int(label)] += len(data.get_patient_subset(idx))

    print(f'Training set\n'
          f'____________\n'
          f'Non-metastatic: {len(shuffled_ones[3:-1])} with {image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[3:-1])} with {image_counts[0]} images.\n'
          f'Total: {len(train_pts)} Patients with {len(train_idx)} images.\n')
    print(f'Evaluation set\n'
          f'______________\n'
          f'Non-metastatic: {len(shuffled_ones[0:3])} with {image_counts[1]} images.\n'
          f'Metastatic: {len(shuffled_zeros[0:3])} with {image_counts[0]} images.\n'
          f'Total: {len(eval_pts)} Patients with {len(eval_idx)} images.\n')
    print(f'Testing set\n'
          f'____________\n'
          f'Non-metastatic: {1} with {image_counts[1]} images.\n'
          f'Metastatic: {1} with {image_counts[0]} images.\n'
          f'Total: {len(test_pts)} Patients with {len(test_idx)} images.\n')

    print(f'Training patients: {train_pts}.\nEvaluation patients: {eval_pts}.\nTest patients: {test_pts}.\n')

    # Create dataloaders for fold
    batch_size = 64
    train_set = torch.utils.data.Subset(data, train_idx)
    eval_set = torch.utils.data.Subset(data, eval_idx)
    test_set = torch.utils.data.Subset(data, eval_idx)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_idx) % batch_size == 1 else False))
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(eval_idx) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=(True if len(eval_idx) % batch_size == 1 else False))

    #####################
    # Prepare model zoo #
    #####################
    # ResNet18
    models = [ResNet18NPlaned(data.shape, start_width=64, n_classes=1)]

    # InceptionResNetV2 Feature Extractor (BigCoMET)
    feature_extractor = inceptionresnetv2(num_classes=1000, pretrained=False)
    classifier = CometClassifierWithBinaryOutput
    models.append(FeatureExtractorToClassifier(data.shape,
                                               feature_extractor=feature_extractor,
                                               classifier=classifier, layer='conv2d_7b'))

    # Xception Feature Extractor
    feature_extractor = xception(num_classes=1000, pretrained=False)
    classifier = torch.nn.Sequential(torch.nn.Linear(2048, 1), torch.nn.Sigmoid())
    models.append(FeatureExtractorToClassifier(data.shape,
                                               feature_extractor=feature_extractor,
                                               classifier=classifier, layer='conv4'))

    # Basic CNNs
    models[len(models):] = [CNNet(data.shape),
                            RegularizedCNNet(data.shape),
                            ParallelCNNet(data.shape),
                            RegularizedParallelCNNet(data.shape)]

    # Put all models on GPU if available
    for model in models:
        if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
            model.to(device)

    ###################
    # Hyperparameters #
    ###################
    epochs = 1500
    learning_rate = 1e-5
    loss_function = nn.BCELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) for model in models]

    ###############
    # Output Prep #
    ###############
    # Prep results file
    for model in models:
        os.makedirs(f'outputs/{model.name}/plots', exist_ok=True)
        with open(f'outputs/{model.name}/results.txt', 'w') as f:
            f.write(f'{model.name} Results\n')
        with open(f'outputs/results.txt', 'w') as f:
            f.write('Overall Results\n')

    # Model path
    try:
        os.mkdir('best_models')
    except FileExistsError:
        pass

    train_loss = [[] for _ in range(len(models))]
    eval_loss = [[] for _ in range(len(models))]
    best_score = [0 for _ in range(len(models))]
    # For each epoch
    for ep in range(epochs):
        print(f'\nEpoch {ep + 1}')
        epoch_loss = [0 for _ in range(len(models))]
        # Train
        for model in models:
            model.train()
        for x, target in train_loader:
            for i, model in enumerate(models):
                out = model(x)
                loss = loss_function(out, target.unsqueeze(1))
                loss.backward()
                epoch_loss[i] += loss.item()
                optimizers[i].step()
        for i, current in enumerate(epoch_loss):
            train_loss[i].append(current / len(train_set))

        # Evaluation
        for i, model in enumerate(models):
            print(f'>>> {model.name}: Training loss: {train_loss[i][-1]}. Evaluation scores:')
            scores = score_model(model, eval_loader, loss_fn=loss_function, print_results=True, make_plot=False,
                                 threshold_type='roc')
            eval_loss[i].append(scores['Loss'])

            with open(f'outputs/{model.name}/results.txt', 'a') as f:
                f.write(f'\n\nEpoch {ep + 1}\n'
                        f'{model.name}: Training loss: {train_loss[i][-1]}. Evaluation scores:')
                for key, item in scores.items():
                    if 'Confusion' not in key:
                        f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|')

            if scores['ROC-AUC'] > best_score[i]:
                best_score[i] = scores['ROC-AUC']
                torch.save(model.state_dict(), f'best_models/Best {model.name}.pth')
                with open(f'outputs/{model.name}/results.txt', 'a') as f:
                    f.write(f'New best {model.name} saved at epoch {ep + 1} with ROC-AUC of {scores["ROC-AUC"]}')
                with open(f'outputs/results.txt', 'a') as f:
                    f.write(f'New best {model.name} saved at epoch {ep + 1} with ROC-AUC of {scores["ROC-AUC"]}')

    with open(f'outputs/results.txt', 'a') as f:
        f.write(f'\nBest ROC-AUC\n')
        for model, score in zip(models, best_score):
            f.write(f'{model.name}: {score:.4f}\n')

    # Testing
    for model in models:
        print(f'>>> {model.name}...')
        model.load_state_dict(torch.load(f'best_models/Best {model.name}.pth'))
        scores, fig = score_model(model, test_loader, print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'outputs/{model.name}_plots.png')
        plt.close(fig)
        with open(f'outputs/{model.name}/results.txt', 'a') as f:
            f.write(f'\n>>> {model.name}...')
            for key, item in scores.items():
                if 'Confusion' not in key:
                    f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
            f.write('_____________________________________________________\n')

    # Plot epoch-wise outputs
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(models):
        plt.plot(range(1, epochs + 1), train_loss[i], label=f'{model.name} Training')
        plt.plot(range(1, epochs + 1), eval_loss[i], label=f'{model.name} Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/losses.png')

# Run
if __name__ == '__main__':
    main()
