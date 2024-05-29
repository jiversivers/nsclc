import torch
import numpy as np


# Create dataloaders
def loader_maker(data, batch_size=32,
                 split=(0.8, 0.15, 0.05),
                 num_workers=(1, 1, 1),
                 shuffle=(True, False, False),
                 drop_last=None,
                 prefetch_factor=None,
                 pin_memory=True):
    # Normalize split and scale to len(data)
    loader_size = [round(x) for x in len(data) * (split / np.sum(split))]
    # Adjust for possible rounding overage
    loader_size[-1] = (loader_size[-1] - 1) if np.sum(loader_size) > len(data) else loader_size[-1]
    print(loader_size)
    # Determine if drop_last is necessary
    if drop_last is None:
        drop_last = ((True if int(sp * len(data)) % batch_size == 1 else False) for sp in split)

    # Create dict for easy function filling where inputs are the same
    kwargs = {'batch_size': batch_size, 'prefetch_factor': prefetch_factor, 'pin_memory': pin_memory}

    sets = torch.utils.data.random_split(
        dataset=data,
        lengths=loader_size)

    loaders = [torch.utils.data.DataLoader(**kwargs, dataset=st, shuffle=sh, num_workers=nw, drop_last=dl)
               for (st, sh, nw, dl) in zip(sets, shuffle, num_workers, drop_last)]

    # handle one-loader edge case, remove iteration necessity
    if len(sets) == 1:
        return loaders[0]
    else:
        return loaders


# Create folds by random sampling then multiplying samples by dataset numbers to augment without repeating slides in
# any group
def fold_augmented_data(data, num_folds=5, augmentation_factor=5):
    # Randomly sample indices from augmented dataset that are _only_ indices of main images
    subsampler = torch.utils.data.SubsetRandomSampler(range(0, len(data), augmentation_factor))

    # Read the subsampler
    indices = [i for i in subsampler]

    data_folds = []
    end = 0
    for fold in range(num_folds):
        # Get end indices for fold of data and set start from where last left off
        start = end
        end = int((fold + 1) * (len(data) / num_folds / augmentation_factor))
        # Get the parent sample indices for the first 1/num_folds samples
        main_parents = indices[start: end]
        # Get the all (number depends on augmentation_factor) the children indices from those indices
        augmented_children = [parent_idx + child_idx for parent_idx in main_parents for child_idx in
                              range(augmentation_factor)]

        # Now actually subset the data
        data_folds.append(torch.utils.data.Subset(data, augmented_children))
    return data_folds
