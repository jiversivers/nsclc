import torch
import numpy as np


# Create dataloaders
def loader_maker(data, batch_size=32,
                 split=(0.8, 0.15, 0.05),
                 num_workers=(1, 1, 1),
                 shuffle=(True, False, False),
                 drop_last=None,
                 prefetch_factor=None):
    # Normalize split
    split = split / np.sum(split)

    # Determine if drop_last is necessary
    if drop_last is None:
        drop_last = ((True if int(sp * len(data)) % batch_size == 1 else False) for sp in split)

    # Create dict for easy function filling where inputs are the same
    kwargs = {'batch_size': batch_size, 'prefetch_factor': prefetch_factor}

    loader_size = [round(sp * len(data)) for sp in split]
    loader_size[-1] = len(data) - np.sum(loader_size[0:-1]) # Correct rounding error causing dataset size mismatch
    sets = torch.utils.data.random_split(
        dataset=data,
        lengths=loader_size)

    loaders = (torch.utils.data.DataLoader(**kwargs, dataset=st, shuffle=sh, num_workers=nw, drop_last=dl)
               for (st, sh, nw, dl) in zip(sets, shuffle, num_workers, drop_last))

    return loaders
