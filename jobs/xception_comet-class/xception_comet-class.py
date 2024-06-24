import os

import torch
from pretrainedmodels import xception
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import torchvision.transforms.v2 as tvt
from my_modules.model_learning.model_metrics import score_model
from my_modules.nsclc import NSCLCDataset, patient_wise_train_test_splitter
from my_modules.custom_models import CometClassifierWithBinaryOutput as Comet, FeatureExtractorToClassifier as FETC


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prep output dirs and files
    os.makedirs('outputs/plots', exist_ok=True)
    with open('outputs/results.txt', 'w') as f:
        f.write('Results')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = xception(num_classes=1000, pretrained=False)

    # Load pretrained from download
    state_dict = torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/xception-43020ad28.pth')
    state_dict['last_linear.weight'] = state_dict.pop('fc.weight')
    state_dict['last_linear.bias'] = state_dict.pop('fc.bias')
    feature_extractor.load_state_dict(state_dict)
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Set up the dataset for this model
    # Images, no mask (feature extractor will hopefully handle this), normalized_to_max (already is),
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=True)
    data.normalize_channels('preset')
    data.transforms = tvt.Compose([tvt.RandomVerticalFlip(p=0.25),
                                   tvt.RandomHorizontalFlip(p=0.25),
                                   tvt.RandomRotation(degrees=(-180, 180))])
    data.augment()
    data.to(device)

    batch_size = 64
    learning_rates = [5e-5, 1e-5, 5e-6, 1e-6]
    optimizer_fns = {'Adam': [torch.optim.Adam, {}]}
    epochs = [125, 250, 500, 1000]
    loss_fn = torch.nn.BCELoss()

    # Define base classifier
    classifier = Comet

    # Prepare data loaders
    train_set, test_set = patient_wise_train_test_splitter(data, n=3)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=(True if len(train_set) % batch_size == 1 else False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                              drop_last=(True if len(test_set) % batch_size == 1 else False))

    training_loss = []
    testing_accuracy = []
    for lr in learning_rates:
        # Make full model
        model = FETC(data.shape, feature_extractor=feature_extractor, classifier=classifier, layer='conv4')
        model.to(device)

        # Make optimizer at the current larning rate with only classifier parameters
        optimizer = optimizer_fns['Adam'][0](model.classifier.parameters(), lr=lr)

        # Nest iteration lists for tracking model learning
        training_loss.append([])
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
                # save trained model at this many epochs
                torch.save(model.state_dict(),
                           f'xception_to_mlp_w_atlas_{ep + 1}-Epochs_{lr}-LearningRate.pth')

                # Test
                scores, figs = score_model(model, test_loader, make_plot=True)
                for key, fig in figs.items():
                    fig.savefig(
                        f'outputs/plots/auc_acc_xception_to_mlp_w_atlas_{key}_{ep + 1}-Epochs_{lr}-LearningRate.png')
                    plt.close(fig)
                with open('outputs/results.txt', 'a') as f:
                    f.write('\n_____________________________________________________\n')
                    for key, item in scores.items():
                        if 'Confusion' not in key:
                            f.write(f'|\t{key:<35} {f'{item:.4f}':>10}\t|\n')
                    f.write('_____________________________________________________\n')

        plt.plot(range(1, 1 + epoch), training_loss[-1])
        plt.savefig(f'outputs/plots/loss_xception_to_mlp_w_atlas_{ep + 1}-Epochs_{lr}-LearningRate.png')
        plt.close()


if __name__ == '__main__':
    main()
