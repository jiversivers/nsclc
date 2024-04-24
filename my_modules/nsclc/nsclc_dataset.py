import torchvision.transforms as transforms
import os
from .image_loaders import load_tiff, load_asc, load_weighted_average, load_bound_fraction
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import cache


class NSCLCDataset:
    def __init__(self, root, mode, xl_file=None, label=None):
        self.root = root
        # Set defaults
        if mode == ['all'] or not mode:
            self.mode = ['orr', 'g', 's', 'photons', 'taumean', 'boundfraction']
        else:
            self.mode = mode
        self.transform = None
        self.augmented = False
        self.dist_transformed = False
        self.normalized = False
        self.label = label
        self.stack_height = len(self.mode)
        self.image_dims = None
        self.scalars = None
        self._name = 'nsclc_'
        self._shape = None

        # Define loading functions for different image types
        load_fn = {'tiff': load_tiff,
                   'asc': load_asc,
                   'weighted_average': load_weighted_average,
                   'ratio': load_bound_fraction}

        # Define a mode dict that matches call to load functions and filename patterns
        self.mode_dict = {'mask': [load_fn['tiff'], os.sep + 'Redox' + os.sep + 'ROI_mask.tiff'],
                          'orr': [load_fn['tiff'], os.sep + 'Redox' + os.sep + 'RawRedoxMap.tiff'],
                          'g': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_phasor_G*'],
                          's': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_phasor_S*'],
                          'photons': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_photons*'],
                          'tau1': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_t1*'],
                          'tau2': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_t2*'],
                          'alpha1': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_a1*'],
                          'alpha2': [load_fn['asc'], os.sep + 'FLIM' + os.sep + '*_a2*']}

        # Find and load features spreadsheet (or load directly if path provided)
        if xl_file is None:
            xl_file = glob.glob(self.root + os.sep + '*.xlsx')
            if not xl_file:
                raise Exception(
                    'Features file not found,'
                    ' input path manually at dataset initialization using xl_file=<path_to_file>.')
            xl_file = xl_file[0]
        self.features = pd.read_excel(xl_file)

        # Prepare a list of FOVs matched to slide names from the features data
        self.fovs_by_slide = [glob.glob(self.root + os.sep + slide + '*') for slide in
                              self.features['Slide Name']]  # This gives a list of lists of fovs sorted by slide
        self.all_fovs = [fov for slide_fovs in self.fovs_by_slide for fov in
                         slide_fovs]  # This will give a list of all fovs (still ordered, but now not nested,
        # making it simple for indexing in __get item__)

        # Iterate through FOVs and check for missing data. If any called modes are missing, pop the entire FOV
        for fov in self.all_fovs[:]:
            for mode in self.mode:
                match mode.lower():
                    case 'taumean':
                        if not (glob.glob(fov + self.mode_dict['alpha1'][1]) and glob.glob(
                                fov + self.mode_dict['tau1'][1])
                                and glob.glob(fov + self.mode_dict['alpha2'][1]) and glob.glob(
                                    fov + self.mode_dict['tau2'][1])):
                            self.all_fovs.remove(fov)
                            break
                    case 'boundfraction':
                        if not (glob.glob(fov + self.mode_dict['alpha1'][1]) and glob.glob(
                                fov + self.mode_dict['alpha2'][1])):
                            self.all_fovs.remove(fov)
                            break
                    case _:
                        if not glob.glob(fov + self.mode_dict[mode][1]):
                            self.all_fovs.remove(fov)
                            break

    # Use property to define name and shape, so that they are automatically updated with latest data setup
    @property
    def name(self):
        self._name = (f"nsclc_{self.label}_{'+'.join(self.mode)}"
                      f'{"_Transformed" if self.dist_transformed else ""}'
                      f'{"_Augmented" if self.augmented else ""}'
                      f'{"_Normalized" if self.normalized else ""}')
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def shape(self):
        self._shape = self[0][0].shape
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def __len__(self):
        if self.augment:
            return len(self.all_fovs)
        else:
            return 5 * len(self.all_fovs)

    @cache
    def __getitem__(self, index):
        # Get image path
        if self.augmented:
            index = int(np.floor(index / 5))  # This will give us the index for the fov
            sub_index = index % 5  # This will give us the index for the crop within the fov
        fov = self.all_fovs[index]

        # Create FOV path dict from mode_dict
        fov_mode_dict = {key: [self.mode_dict[key][0], fov + self.mode_dict[key][1]] for key in self.mode_dict.keys()}
        fov_mode_dict['boundfraction'] = [load_bound_fraction, [fov_mode_dict['alpha1'], fov_mode_dict['alpha2']]]
        fov_mode_dict['taumean'] = [load_weighted_average,
                                    [fov_mode_dict['alpha1'], fov_mode_dict['tau1'], fov_mode_dict['alpha2'],
                                     fov_mode_dict['tau2']]]

        # Load mask
        fov_mask = fov_mode_dict['mask'][0](fov_mode_dict['mask'])
        fov_mask[fov_mask == 0] = float('nan')

        # Preallocate output tensor based on mask size
        self.image_dims = (self.stack_height,) + tuple(fov_mask.size()[1:])
        X = torch.empty(self.image_dims, dtype=torch.float32)

        # Load modes using load functions
        for ii, mode in enumerate(self.mode):
            X[ii] = fov_mode_dict[mode][0](fov_mode_dict[mode])

        # Scale by the max value of normalized
        if self.normalized:
            X = X / self.scalars

        # Crop and sub index if necessary
        if self.augmented:
            cropper = transforms.FiveCrop((round(self.image_dims[1] / 2), round(self.image_dims[2] / 2)))
            X = cropper(X)
            fov_mask = cropper(fov_mask)
            X = X[sub_index]
            fov_mask = fov_mask[sub_index]

        # Get data label and apply mask to all channels for binary classes
        slide_idx = [fov in slide for slide in self.fovs_by_slide].index(True)  # Get features index
        match self.label.lower():
            case 'response' | 'r':
                X[torch.isnan(fov_mask).expand(X.size(0), *fov_mask.size()[1:])] = float('nan')
                y = 1 if self.features['Status (NR/R)'].iloc[slide_idx] == 'R' else 0
            case 'metastases' | 'mets' | 'm':
                X[torch.isnan(fov_mask).expand(X.size(0), *fov_mask.size()[1:])] = float('nan')
                y = 1 if self.features['Status (Mets/NM)'].iloc[slide_idx] == 'NM' else 0
            case 'mask':
                y = fov_mask
            case _:
                raise Exception('No data label selected. Update label attribute of dataset and try again.')

        # Apply distribution transform, if called
        if self.dist_transformed:
            X_dist = torch.empty((self.stack_height,) + (self.nbins,), dtype=torch.float32)
            for ch, mode in enumerate(X):
                X_dist[ch], _ = torch.histogram(mode, bins=self.nbins, range=[0, 1], density=True)
            X = X_dist

        return X, y

    def dist_transform(self, nbins=25):
        self.nbins = nbins
        if not self.normalized:
            print(
                'Normalization is automatically applied for the distribution transform.\n     '
                'This can be manually overwritten by setting the NORMALIZED attribute to False')
            self.normalize_channels_to_max()
        self.dist_transformed = True

    def show_random(self):
        if self.dist_transformed:
            _, ax = plt.subplots(5, 1, figsize=(10, 10))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                ax[ii].plot(self[index][0].T, label=self.mode)
                ax[ii].legend()
                ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                ax[ii].set_title(f'Label: {self[index][1]}', fontsize=10)
        else:
            transform = transforms.ToPILImage()
            _, ax = plt.subplots(5, len(self.mode), figsize=(10, 10))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                img = self[index][0]
                img[torch.isnan(img)] = 0
                lab = self[index][1]
                if len(self.mode) > 1:
                    for jj in range(len(self.mode)):
                        ax[ii, jj].imshow(transform(img[jj]))
                        ax[ii, jj].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                               labelbottom=False)
                        ax[ii, jj].set_title(f'Response: {self[index][1]}. \n Mode: {self.mode[jj]}', fontsize=10)
                else:
                    ax[ii].imshow(transform(self[index][0][jj]))
                    ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                       labelbottom=False)
                    ax[ii].set_title(f'Response: {lab}. \n Mode: {self.mode[jj]}', fontsize=10)
        plt.show()

    def augment(self):
        self.augmented = True

    def normalize_channels_to_max(self):
        # Find the max for each mode across the entire dataset
        # This is mildly time-consuming, so we only do it once, then store the scalar and mark the set as normalized.
        # In order to make distributions consistent, this step will be required for dist transforms, so it will be
        # checked before performing the transform

        # Check if previously normalized
        if self.normalized:
            return

        # Temporarily turn off augmentation so max doesn't have to run 5x
        temp_aug = self.augmented  # Store for restoring later
        self.augmented = False

        # Preallocate an array. Each row is an individual image, each column is mode
        maxes = np.zeros((len(self), self.stack_height), dtype=np.float32)
        for ii, (stack, _) in enumerate(self):
            maxes[ii] = np.nanmax(stack, axis=(1, 2))
        self.scalars = np.max(maxes, axis=0)
        self.scalars = self.scalars[:, None, None]  # Broadcast for easy channel-wise scaling

        # Set normalized to TRUE so images will be scaled to max when retrieved
        self.normalized = True

        # Reset augmented
        self.augmented = temp_aug
