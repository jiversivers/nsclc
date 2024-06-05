import torchvision.transforms as t
import os
from .helper_functions import load_tiff, load_asc, load_weighted_average, load_bound_fraction, convert_mp_to_torch
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import glob
import multiprocessing as mp
import ctypes
from torch.utils.data import Dataset


class NSCLCDataset(Dataset):
    """
    NSCLC dataset class to load NSCLC images from root dir (arg). When __getitem__ is used, the dataset will return a
    tuple with the fov stack of modes (ordered by the input, or default order with 'all') at index (arg) and the binary
    label of the sample, or the mask if 'mask' is set for the label. IN the binary classes, the label will be 0 for
    non-responders or positive metastases and 1 otherwise. As a shorthand, a positive class (1) is the positive outcome.

    Image modes are specified at creation and can include any typical MPM image modes available from the dataset and
    derived modes (bound fraction and mean lifetime).

    Attributes:
        - root (string): Root directory of NSCLC dataset.
        - mode (list): Ordered list of image modes as they will be returned.
        - label (string): Name of feature to be returned as data label.
        - stack_height (int): Height of image stack, equivalent to number of modes
        - image_dims (tuple): Tuple of image dimensions.
        - scalars (arr): Max value of each mode used in normalization
        - name (string): Name of the dataset dependent on data parameters.
        - shape (tuple): Shape of individual image stack
        - reset_cache (callable): Function to reset cached data in shared caches
        - dist_transform (callable): Applies distribution transform to image stack
        - dist_transformed (bool): Returns whether image stack is histograms
        - augment (callable): Applies image augmentation to dataset using FiveCrop augmentation
        - augmented (bool): Returns whether dataset is augmented
        - normalize_channels_to_max (callable): Applies normalization to image stack channel-wise
        - normalized (bool): Returns whether image stack is normalized
        - show_random (callable): Shows 5 random samples from dataset
        - device (str or torch.device): Device type or device ID
        - to (callable): Move any currently cached items to DEVICE from input argument in call and return all future
            items on DEVICE.
    """

    # region Main Dataset Methods (init, len, getitem)
    def __init__(self, root, mode, xl_file=None, label=None, mask_on=True, transforms=None,
                 device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))):
        self.transforms = transforms
        self.root = root
        self.device = device

        # Set hidden property defaults
        self._name = 'nsclc_'
        self._shape = None
        self._augmented = False
        self._dist_transformed = False
        self._psuedo_rgb = False
        self._normalized = False
        self._nbins = 25

        # Set defaults
        self.mode = mode
        self.label = label
        self.mask_on = mask_on

        # Set data descriptors
        self.stack_height = len(self.mode)
        self.image_dims = None
        self.scalars = None

        # Init placeholder cache arrays
        self.index_cache = None
        self.shared_x = None
        self.shared_y = None

        # Define loading functions for different image types
        load_fn = {'tiff': load_tiff,
                   'asc': load_asc,
                   'weighted_average': load_weighted_average,
                   'ratio': load_bound_fraction}

        # Define a mode dict that matches call to load functions and filename regex
        self.mode_dict = {'mask': [load_fn['tiff'], rf'.*?(\\|/)Redox(\\|/)ROI_mask\.(tiff|TIFF)'],
                          'orr': [load_fn['tiff'], rf'.*?(\\|/)Redox(\\|/)RawRedoxMap\.(tiff|TIFF)'],
                          'g': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_phasor_G\.(asc|ASC)'],
                          's': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_phasor_S\.(asc|ASC)'],
                          'photons': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_photons\.(asc|ASC)'],
                          'tau1': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_t1\.(asc|ASC)'],
                          'tau2': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_t2\.(asc|ASC)'],
                          'alpha1': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_a1\.(asc|ASC)'],
                          'alpha2': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_a2\.(asc|ASC)']}
        self.mode_dict = {key: [item[0], re.compile(item[1])] for key, item in self.mode_dict.items()}

        # Find and load features spreadsheet (or load directly if path provided)
        if xl_file is None:
            xl_file = glob.glob(self.root + os.sep + '*.xlsx')
            if not xl_file:
                raise Exception(
                    'Features file not found,'
                    ' input path manually at dataset initialization using xl_file=<path_to_file>.')
            self.xl_file = xl_file[0]
        self.features = pd.read_excel(self.xl_file)

        # Prepare a list of FOVs from data dir matched to slide names from the features excel file
        self.fovs_by_slide = [glob.glob(self.root + os.sep + slide + '*') for slide in
                              self.features['Slide Name']]  # This gives a list of lists of fovs sorted by slide
        self.all_fovs = [fov for slide_fovs in self.fovs_by_slide for fov in
                         slide_fovs]  # This will give a list of all fovs (still ordered, but now not nested,
        # making it simple for indexing in __getitem__)

        # Make and indexed FOV-LUT dict list of loaders and files
        self.fov_mode_dict = [{} for _ in range(len(self.all_fovs))]
        # Iterate through all FOVs
        for index, fov in enumerate(self.all_fovs):
            # Iterate through all base modes (from mode_dict)
            for mode, (load_fn, file_pattern) in self.mode_dict.items():
                matched = []
                # Iterate through the current FOV tree
                for trunk, dirs, files in os.walk(fov):
                    # Check if any file in the tree matches the pattern for the mode from the base LUT (mode_dict)
                    for file in files:
                        matched.append(re.match(file_pattern, os.path.join(trunk, file)))
                # If exactly one file matched, then add it to the FOV-LUT dict
                if sum(t is not None for t in matched) == 1:
                    for match in matched:
                        if match:
                            self.fov_mode_dict[index][mode] = [load_fn, match.string]
                # Else, add <None> for later removal and move on to next FOV
                else:
                    self.fov_mode_dict[index][mode] = [fov, None]

            # Add derived modes
            self.fov_mode_dict[index]['boundfraction'] = [load_bound_fraction, [self.fov_mode_dict[index]['alpha1'],
                                                                                self.fov_mode_dict[index]['alpha2']]]
            self.fov_mode_dict[index]['taumean'] = [load_weighted_average,
                                                    [self.fov_mode_dict[index]['alpha1'],
                                                     self.fov_mode_dict[index]['tau1'],
                                                     self.fov_mode_dict[index]['alpha2'],
                                                     self.fov_mode_dict[index]['tau2']]]
        # Remove items that are missing a called mode
        # Note the [:] makes a copy of the list so indices don't change on removal
        for ii, fov_lut in enumerate(self.fov_mode_dict[:]):
            for mode in self.mode:
                match mode.lower():
                    case 'taumean':
                        if not all([fov_lut['alpha1'][1], fov_lut['tau1'][1],
                                    fov_lut['alpha2'][1], fov_lut['tau2'][1]]):
                            self.all_fovs.remove(fov_lut['alpha1'][0])
                            self.fov_mode_dict.remove(fov_lut)
                            break
                    case 'boundfraction':
                        if not all([fov_lut['alpha1'][1], fov_lut['alpha2'][1]]):
                            self.all_fovs.remove(fov_lut['alpha1'][0])
                            self.fov_mode_dict.remove(fov_lut)
                            break
                    case _:
                        if fov_lut[mode][1] is None:
                            self.all_fovs.remove(fov_lut[mode][0])
                            self.fov_mode_dict.remove(fov_lut)
                            break

    def __len__(self):
        if self.augmented:
            return 5 * len(self.all_fovs)
        else:
            return len(self.all_fovs)

    def __getitem__(self, index):
        # Check if indexed sample is in cache (by checking for index in index_cache)...
        # if it is, pull it from the cache;
        if (all(cache_arr is not None for cache_arr in [self.index_cache, self.shared_x, self.shared_y])
                and index in self.index_cache):
            x = self.shared_x[index]
            y = self.shared_y[index]
            return x, y

        # load the sample, and cache the sample (if cache is open)
        # region Load Data and Label
        # Get image path from index
        if self.augmented:
            fov_index = int(np.floor(index / 5))  # This will give us the index for the fov
            sub_index = index % 5  # This will give us the index for the crop within the fov
        else:
            fov_index = index
        fov = self.all_fovs[fov_index]

        # Load mask
        fov_mask = self.fov_mode_dict[fov_index]['mask'][0](self.fov_mode_dict[fov_index]['mask']).to(self.device)
        fov_mask[fov_mask == 0] = float('nan')

        # Preallocate output tensor based on mask size
        self.image_dims = (self.stack_height,) + tuple(fov_mask.size()[1:])
        x = torch.empty(self.image_dims, dtype=torch.float32, device=self.device)

        # Load modes using load functions
        for ii, mode in enumerate(self.mode):
            x[ii] = self.fov_mode_dict[fov_index][mode][0](self.fov_mode_dict[fov_index][mode]).to(self.device)

        # Scale by the max value of normalized
        if self.normalized:
            x = x / self.scalars

        # Crop and sub index if necessary
        if self.augmented:
            cropper = t.FiveCrop((round(self.image_dims[1] / 2), round(self.image_dims[2] / 2)))
            x = cropper(x)
            fov_mask = cropper(fov_mask)
            x = x[sub_index]
            fov_mask = fov_mask[sub_index]

        # Unsqueeze so "color" is dim 1 and expand to look like an RGB image
        # New image dims: (M, C, H, W), where M is the mode, C is the psuedo-color channel, H and W are height and width
        if self.psuedo_rgb:
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)

        # Get data label and apply mask to all channels for binary classes
        # Get features based on what slide the FOV is from
        slide_idx = [fov in slide for slide in self.fovs_by_slide].index(True)
        match self.label:
            case 'Response':
                if self.mask_on:
                    x[torch.isnan(fov_mask).expand(x.size(0), *fov_mask.size()[1:])] = float('nan')
                y = torch.tensor(1 if self.features['Status (NR/R)'].iloc[slide_idx] == 'R' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Metastases':
                if self.mask_on:
                    x[torch.isnan(fov_mask).expand(x.size(0), *fov_mask.size()[1:])] = float('nan')
                y = torch.tensor(1 if self.features['Status (Mets/NM)'].iloc[slide_idx] == 'NM' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Mask':
                y = fov_mask
            case None:
                y = torch.tensor(-999999,  # Placeholder for NaN label
                                 dtype=torch.float32, device=self.device)
            case _:
                raise Exception('An unrecognized label is in use. Update label attribute of dataset and try again.')

        # Apply distribution transform, if called
        if self.dist_transformed:
            x_dist = torch.empty((self.stack_height,) + (self._nbins,), dtype=torch.float32, device=self.device)
            for ch, mode in enumerate(x):
                x_dist[ch], _ = torch.histogram(mode.cpu(), bins=self._nbins, range=[0, 1], density=True)
            x = x_dist

        # Cache the sample if the caches have been opened
        if self.index_cache is None:
            self._open_cache(x, y)
        self.shared_x[index] = x.to(self.device)
        self.shared_y[index] = y.to(self.device)
        self.index_cache[index] = index

        # Apply transforms that were input (if any)
        x = self.transforms(x) if self.transforms is not None else x
        return x, y
        # endregion

    def reset_cache(self):
        print('Cache reset')
        # Hard reset
        self.index_cache = None
        self.shared_x = None
        self.shared_y = None
        self._shape = None

    def _open_cache(self, x, y):
        # Setup shared memory arrays (i.e. caches that are compatible with multiple workers)
        # negative initialization ensure no overlap with actual cached indices
        index_cache_base = mp.Array(ctypes.c_int, len(self) * [-1])
        shared_x_base = mp.Array(ctypes.c_float, int(len(self) * np.prod(x.shape)))

        # Label-size determines cache size, so if no label is set, we will fill cache with -999999 at __getitem__
        match self.label:
            case 'Response' | 'Metastases' | None:
                shared_y_base = mp.Array(ctypes.c_float, len(self) * [-1])
                y_shape = (1,)
            case 'Mask':
                shared_y_base = mp.Array(ctypes.c_float, int(len(self) * np.prod(y.shape)))
                y_shape = tuple(y.shape)
            case _:
                raise Exception('An unrecognized label is in use that is blocking the cache from initializing. '
                                'Update label attribute of dataset and try again.')

        # Convert all arrays to desired data structure
        self.index_cache = convert_mp_to_torch(index_cache_base, (len(self),), device=self.device)
        self.shared_x = convert_mp_to_torch(shared_x_base, (len(self),) + x.shape, device=self.device)
        self.shared_y = convert_mp_to_torch(shared_y_base, (len(self),) + y_shape, device=self.device)
        print('Cache opened')

    def to(self, device):
        # Move caches to device
        self.index_cache = self.index_cache.to(device)
        for i, idx in enumerate(self.index_cache):
            self.index_cache[i] = idx.to(device)
        self.shared_x = self.shared_x.to(device)
        for i, x in enumerate(self.shared_x):
            self.shared_x[i] = x.to(device)
        self.shared_y = self.shared_y.to(device)
        for i, y in enumerate(self.shared_y):
            self.shared_y[i] = y.to(device)

        # Move any self-held tensors to device for ops compatibility
        self.scalars = self.scalars.to(device) if self.scalars is not None else None

        # Update device for future items
        self.device = device

    # endregion

    # region Properties
    # Name, shape, label, mode, mask
    # Use of property (instead of simple attribute) to define ensures automatic updates with latest data setup and/or
    # appropriate clearing of the cache

    # Name (cannot be changed directly)
    @property
    def name(self):
        self._name = (f"nsclc_{self.label}_{'+'.join(self.mode)}"
                      f'{"_Histogram" if self.dist_transformed else ""}'
                      f'{"_Augmented" if self.augmented else ""}'
                      f'{"_Normalized" if self.normalized else ""}'
                      f'{"_Masked" if self.mask_on else ""}')
        return self._name

    # Shape (cannot be changed directly)
    @property
    def shape(self):
        if self._shape is None:
            self._shape = self[0][0].shape
        return self._shape

    # Label
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        match label.lower():
            case 'response' | 'r':
                label = 'Response'
            case 'metastases' | 'mets' | 'm':
                label = 'Metastases'
            case 'mask':
                label = 'Mask'
                self.mask_on = False
            case _:
                raise Exception('Invalid data label entered. Allowed labels are "RESPONSE", "METASTASES", and "MASK".')
        if hasattr(self, '_label') and label != self.label:
            self.reset_cache()
        self._label = label

    # Modes
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # Set default/shortcut behavior
        if mode == ['all'] or mode == 'all' or mode is None:
            mode = ['orr', 'g', 's', 'photons', 'taumean', 'boundfraction']
        # If this is not the __init__ run
        if hasattr(self, '_mode') and mode != self.mode:
            # A mode update is essentially a new dataset, so we will re-init it, but we want to make all the other
            # aspects match, so we check and store them first. Once the dataset is reinitialized, we can reset the
            # other properties.
            temp_aug = self.augmented
            temp_norm = self.normalized
            temp_dist = self.dist_transformed

            # Re-init with new modes
            self.__init__(self.root, mode, xl_file=self.xl_file, label=self.label, mask_on=self.mask_on,
                          transforms=self.transforms, device=self.device)

            # Reset properties
            self.normalized = temp_norm
            self.augmented = temp_aug
            self.dist_transformed = temp_dist
        # If this is the __init__ run
        else:
            self._mode = mode

    # Masking
    @property
    def mask_on(self):
        return self._mask_on

    @mask_on.setter
    def mask_on(self, mask_on):
        if hasattr(self, '_mask_on') and mask_on != self.mask_on:
            self.reset_cache()
        self._mask_on = mask_on

    # endregion

    # region Distribution transform
    def dist_transform(self, nbins=25):
        # If it is already transformed to the same bin number, leave it alone
        if self.dist_transformed and nbins == self._nbins:
            pass
        # If something is changed, reset and update
        else:
            self.reset_cache()
            self._nbins = nbins
            if not self.normalized:
                print(
                    'Normalization is automatically applied for the distribution transform.\n     '
                    'This can be manually overwritten by setting the NORMALIZED attribute to False')
                self.normalize_channels_to_max()
        self.dist_transformed = True

    @property
    def dist_transformed(self):
        return self._dist_transformed

    @dist_transformed.setter
    def dist_transformed(self, dist_transformed):
        # If it is changing, reset and update
        if dist_transformed is not self.dist_transformed:
            self._dist_transformed = dist_transformed
            self.reset_cache()

    # endregion

    # region Augmentation
    def augment(self):
        self.augmented = True

    @property
    def augmented(self):
        return self._augmented

    @augmented.setter
    def augmented(self, augmented):
        # If it is changing, reset and update
        if augmented is not self.augmented:
            self._augmented = augmented
            self.reset_cache()

    # endregion

    # region Psuedo-RGB
    def transform_to_psuedo_rgb(self):
        # Update the property
        self.psuedo_rgb = True

    @property
    def psuedo_rgb(self):
        return self._psuedo_rgb

    @psuedo_rgb.setter
    def psuedo_rgb(self, psuedo_rgb):
        # If it is changing, reset and update
        if psuedo_rgb is not self.psuedo_rgb:
            self._psuedo_rgb = psuedo_rgb
            self.reset_cache()

    # endregion

    # region Normalization
    def normalize_channels_to_max(self):
        # Find the max for each mode across the entire dataset
        # This is mildly time-consuming, so we only do it once, then store the scalar and mark the set as normalized.
        # In order to make distributions consistent, this step will be required for dist transforms, so it will be
        # checked before performing the transform

        # Temporarily turn psuedo_rgb off (if on) so we can save 2/3 memory and not ahve to worry about dim shifts for
        # both cases
        temp_psuedo = self.psuedo_rgb
        self.psuedo_rgb = False

        # Preallocate an array. Each row is an individual image, each column is mode
        maxes = torch.zeros(len(self), self.stack_height, dtype=torch.float32, device=self.device)
        for ii, (stack, _) in enumerate(self):
            # Does the same as np.nanmax(stack, dim=(1,2)) but keeps the tensor on the GPU
            maxes[ii] = torch.max(torch.max(torch.nan_to_num(stack, nan=-100000), 1).values, 1).values
        self.scalars = torch.max(maxes, 0).values
        self.scalars = self.scalars.unsqueeze(1).unsqueeze(2).to(self.device)

        # Reset psuedo_rgb
        self.psuedo_rgb = temp_psuedo

        # Set normalized to TRUE so images will be scaled to max when retrieved
        self._normalized = True

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, normalized):
        # Check if the cache needs to be reset
        if normalized is not self.normalized:
            self._normalized = normalized
            self.scalars = None
            self.reset_cache()
        # Apply the normalization requested (note: method call will set attribute if TRUE)
        if normalized:
            self.normalize_channels_to_max()

    # endregion

    # region Show random data samples
    def show_random(self):
        if self.dist_transformed:
            _, ax = plt.subplots(5, 1, figsize=(10, 10))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                ax[ii].plot(self[index][0].T, label=self.mode)
                ax[ii].legend()
                ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                   labelbottom=False)
                ax[ii].set_title(f'{self.label}: {self[index][1]}', fontsize=10)
        else:
            transform = t.ToPILImage()
            _, ax = plt.subplots(5, len(self.mode), figsize=(10, 10))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                img = self[index][0]
                img[torch.isnan(img)] = 0
                lab = self[index][1]
                if len(self.mode) > 1:
                    for jj in range(len(self.mode)):
                        ax[ii, jj].imshow(transform(img[jj]))
                        ax[ii, jj].tick_params(top=False, bottom=False, left=False, right=False,
                                               labelleft=False,
                                               labelbottom=False)
                        ax[ii, jj].set_title(f'{self.label}: {self[index][1]}. \n Mode: {self.mode[jj]}',
                                             fontsize=10)
                else:
                    ax[ii].imshow(transform(self[index][0]))
                    ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                       labelbottom=False)
                    ax[ii].set_title(f'{self.label}: {lab}. \n Mode: {self.mode[0]}', fontsize=10)
        plt.show()

    # endregion
