import torch
from pretrainedmodels import inceptionresnetv2
import torch.multiprocessing as mp

from my_modules.model_learning.loader_maker import fold_augmented_data
from my_modules.nsclc import NSCLCDataset
from my_modules.custom_models import CometClassifier, FeatureExtractorToClassifier as FETC


def main():
    # Set up multiprocessing context
    mp.set_start_method('forkserver', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our base feature extractor and turn the gradients off -- we won't train it, just use it to feed our MLP.
    feature_extractor = inceptionresnetv2(num_classes=1001, pretrained=False)

    # Load pretrained from download
    feature_extractor.load_state_dict(
        torch.load(r'/home/jdivers/data/torch_checkpoints/pretrained_models/inceptionresnetv2-520b38e4.pth'))
    for params in feature_extractor.parameters():
        params.requires_grad = False

    # Set up the dataset for this model
    # Images, no mask (feature extractor will hopefully handle this), normalized (already is),
    data = NSCLCDataset('data/NSCLC_Data_for_ML', ['orr', 'taumean', 'boundfraction'], device='cpu',
                        label='Metastases', mask_on=False)
    print('Normalizing data to channel max...')
    data.augment()
    data.normalize_channels_to_max()

    batch_size = 32
    learning_rates = [0.1, 0.01, 0.001, 0.00001]
    optimizer_fns = {'SGD': [torch.optim.SGD, {'momentum': 0.9}],
                     'RMSprop': [torch.optim.RMSprop, {'momentum': 0.9}],
                     'Adam': [torch.optim.Adam, {}]}
    epochs = [125, 250, 500, 100, 2500]
    loss_fn = torch.nn.CrossEntropyLoss()

    # Dry run feature extractor to get output dims for classifier creation
    feature_extractor.eval()
    with torch.no_grad():
        x = torch.rand(1, *data.shape)
        dry_run = feature_extractor.features(x)
    feature_map_dims = dry_run.shape

    # Define base extractor
    classifier = CometClassifier(feature_map_dims[1:])

    # Send both submodules to device
    feature_extractor.to(device)
    classifier.to(device)

    models = []
    accuracy = []
    running_loss = []

    data_folds = fold_augmented_data(data, num_folds=5, augmentation_factor=5)

    for _ in range(2):
        data.mask_on = not data.mask_on  # just flip it
        for optim_name, (optim_fn, settings) in optimizer_fns.items():
            for lr in learning_rates:
                for epoch in epochs:
                    for fold, test_set in enumerate(data_folds):
                        print(f'Fold {fold + 1}\n________________________________')
                        # Make model, loaders, & optimizer for fold
                        model = FETC(data.shape, feature_extractor=feature_extractor, classifier=classifier,
                                     layer='conv2d_7b')
                        model.to(device)
                        train_sets = [data_folds[index] for index in range(5) if index != fold]
                        train_set = torch.utils.data.ConcatDataset(train_sets)
                        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
                        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
                        optimizer = optim_fn(model.parameters(), lr=lr, **settings)

                        # Train
                        model.train()
                        running_loss.append([])
                        for ep in range(epoch):
                            for x, target in train_loader:
                                if torch.cuda.is_available() and not x.is_cuda:
                                    x = x.cuda()
                                if torch.cuda.is_available() and not target.is_cuda:
                                    target = target.cuda()
                                # Extract features
                                out = model(x)

                                # Array to reform target to look like out
                                y = torch.zeros_like(out)

                                # Put 100% certainty on the class
                                for r, t in enumerate(target):
                                    y[r, t.long()] = 1.

                                loss = loss_fn(out, y)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                running_loss[fold].append(loss.item())

                        # save trained fold
                        models.append(model)
                        torch.save(model.state_dict(), f'big-comet_Fold{fold + 1}.pth')

                        # Test
                        correct = 0
                        model.eval()
                        for x, target in test_loader:
                            if torch.cuda.is_available() and not x.is_cuda:
                                x = x.cuda()
                            if torch.cuda.is_available() and not target.is_cuda:
                                target = target.cuda()
                            out = model(x)
                            pred = torch.argmax(out, dim=1)
                            correct += torch.sum(pred == target).item()
                        accuracy.append(100 * correct / len(test_loader.sampler))
                    print(f'Mask: {data.mask_on} -- Optim: {optim_name} -- LR: {lr} -- Epoch: {epoch}')
                    print(f'>>>Inception + MLP average accuracy: {sum(accuracy) / len(accuracy):.2f}%<<<')

if __name__ == '__main__':
    main()
