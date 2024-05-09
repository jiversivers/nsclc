import torchvision.transforms as transforms
import os
from .image_loaders import load_tiff, load_asc, load_weighted_average, load_bound_fraction
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import lru_cache
import re
import glob

class NSCLCDataset:
    """
    NSCLC dataset class to load NSCLC images from root dir (arg).

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
        - dist_transform (callable): Applies distribution transform to image stack
        - dist_transformed (bool): Returns whether image stack is histograms
        - augment (callable): Applies image augmentation to dataset using FiveCrop augmentation
        - augmented (bool): Returns whether dataset is augmented
        - normalize_channels_to_max (callable): Applies normalization to image stack channel-wise
        - normalized (bool): Returns whether image stack is normalized
        - show_random (callable): Shows 5 random samples from dataset
    """

    def __init__(self, root, mode, xl_file=None, label=None):
        self.root = root
        # Set defaults
        if mode == ['all'] or not mode:
            self.mode = ['orr', 'g', 's', 'photons', 'taumean', 'boundfraction']
        else:
            self.mode = mode
        self.label = label
        self.stack_height = len(self.mode)
        self.image_dims = None
        self.scalars = None
        self.fov_mode_dict = []

        self._name = 'nsclc_'
        self._shape = None
        self._augmented = False
        self._dist_transformed = False
        self._normalized = False
        self._nbins = 25

        # Define loading functions for different image types
        load_fn = {'tiff': load_tiff,
                   'asc': load_asc,
                   'weighted_average': load_weighted_average,
                   'ratio': load_bound_fraction}

        # Define a mode dict that matches call to load functions and filename regex
        self.mode_dict = {'mask': [load_fn['tiff'], rf'.*?\{os.sep}Redox\{os.sep}ROI_mask\.(tiff|TIFF)'],
                          'orr': [load_fn['tiff'], rf'.*?\{os.sep}Redox\{os.sep}RawRedoxMap\.(tiff|TIFF)'],
                          'g': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_phasor_G.*?\.(asc|ASC)'],
                          's': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_phasor_S.*?\.(asc|ASC)'],
                          'photons': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_photons.*?\.(asc|ASC)'],
                          'tau1': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_t1.*?\.(asc|ASC)'],
                          'tau2': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_t2.*?\.(asc|ASC)'],
                          'alpha1': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_a1.*?\.(asc|ASC)'],
                          'alpha2': [load_fn['asc'], rf'.*?\{os.sep}FLIM\{os.sep}.*?_a2.*?\.(asc|ASC)']}
        self.mode_dict = {key: [item[0], re.compile(item[1])] for key, item in self.mode_dict.items()}

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
        # making it simple for indexing in __getitem__)

        # Make and indexed FOV-LUT dict list of loaders and files
        # Iterate through all FOVs
        for index, fov in enumerate(self.all_fovs):
            # Iterate through all base modes (from mode_dict)
            for mode, (load_fn, file_pattern) in self.mode_dict.keys():
                matched = []
                # Iterate through the current FOV tree
                for root, _, files in os.walk(fov):
                    # Check if any file in the tree matches the pattern for the mode from the base LUT (mode_dict)
                    for file in files:
                        matched.append(re.match(file_pattern, os.path.join(root, file)))
                # If exactly one file matched, then add it to the FOV-LUT dict
                if sum(t is not None for t in matched) == 1:
                    self.fov_mode_dict[index][mode] = [load_fn, next(match.string for match in matched)]
                # Else, add <None> for later removal and move on to next FOV
                else:
                    self.fov_mode_dict[index][mode] = [fov, None]
                    break

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

    @lru_cache()
    def __getitem__(self, index):
        # Get image path
        if self.augmented:
            index = int(np.floor(index / 5))  # This will give us the index for the fov
            sub_index = index % 5  # This will give us the index for the crop within the fov
        fov = self.all_fovs[index]

        # Load mask
        fov_mask = self.fov_mode_dict[index]['mask'][0](self.fov_mode_dict[index]['mask'])
        fov_mask[fov_mask == 0] = float('nan')

        # Preallocate output tensor based on mask size
        self.image_dims = (self.stack_height,) + tuple(fov_mask.size()[1:])
        x = torch.empty(self.image_dims, dtype=torch.float32)

        # Load modes using load functions
        for ii, mode in enumerate(self.mode):
            x[ii] = self.fov_mode_dict[index][mode][0](self.fov_mode_dict[index][mode])

        # Scale by the max value of normalized
        if self.normalized:
            x = x / self.scalars

        # Crop and sub index if necessary
        if self.augmented:
            cropper = transforms.FiveCrop((round(self.image_dims[1] / 2), round(self.image_dims[2] / 2)))
            x = cropper(x)
            fov_mask = cropper(fov_mask)
            x = x[sub_index]
            fov_mask = fov_mask[sub_index]

        # Get data label and apply mask to all channels for binary classes
        slide_idx = [fov in slide for slide in self.fovs_by_slide].index(True)  # Get features index
        match self.label.lower():
            case 'response' | 'r':
                x[torch.isnan(fov_mask).expand(x.size(0), *fov_mask.size()[1:])] = float('nan')
                y = 1 if self.features['Status (NR/R)'].iloc[slide_idx] == 'R' else 0
            case 'metastases' | 'mets' | 'm':
                x[torch.isnan(fov_mask).expand(x.size(0), *fov_mask.size()[1:])] = float('nan')
                y = 1 if self.features['Status (Mets/NM)'].iloc[slide_idx] == 'NM' else 0
            case 'mask':
                y = fov_mask
            case _:
                raise Exception('No data label selected. Update label attribute of dataset and try again.')

        # Apply distribution transform, if called
        if self.dist_transformed:
            x_dist = torch.empty((self.stack_height,) + (self._nbins,), dtype=torch.float32)
            for ch, mode in enumerate(x):
                x_dist[ch], _ = torch.histogram(mode, bins=self._nbins, range=[0, 1], density=True)
            x = x_dist

        return x, y

    ##############
    # Properties #
    ##############
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

    ##########################
    # Distribution transform #
    ##########################
    def dist_transform(self, nbins=25):
        # If it is already transformed to the same bin number, leave it alone
        if self.dist_transformed and nbins == self._nbins:
            pass
        # If something is changed, reset and update
        else:
            self.__getitem__.cache_clear()
            self._nbins = nbins
            self.dist_transformed = True
            if not self.normalized:
                print(
                    'Normalization is automatically applied for the distribution transform.\n     '
                    'This can be manually overwritten by setting the NORMALIZED attribute to False')
                self.normalize_channels_to_max()

    @property
    def dist_transformed(self):
        return self._dist_transformed

    @dist_transformed.setter
    def dist_transformed(self, dist_transformed):
        # If it is changing, reset and update
        if dist_transformed is not self.dist_transformed:
            self.__getitem__.cache_clear()
            self._dist_transformed = dist_transformed

    ################
    # Augmentation #
    ################
    def augment(self):
        self.augmented = True

    @property
    def augmented(self):
        return self._augmented

    @augmented.setter
    def augmented(self, augmented):
        # If it is changing, reset and update
        if augmented is not self.augmented:
            self.__getitem__.cache_clear()
            self._augmented = augmented

    #################
    # Normalization #
    #################
    def normalize_channels_to_max(self):
        # Find the max for each mode across the entire dataset
        # This is mildly time-consuming, so we only do it once, then store the scalar and mark the set as normalized.
        # In order to make distributions consistent, this step will be required for dist transforms, so it will be
        # checked before performing the transform

        # Check if previously normalized
        if self._normalized:
            return

        # If scalars have not been previously calculated, calculate
        if self.scalars is None:
            # Preallocate an array. Each row is an individual image, each column is mode
            maxes = np.zeros((len(self), self.stack_height), dtype=np.float32)
            for ii, (stack, _) in enumerate(self):
                maxes[ii] = np.nanmax(stack, axis=(1, 2))
            self.scalars = np.max(maxes, axis=0)
            self.scalars = self.scalars[:, None, None]  # Broadcast for easy channel-wise scaling

        # Set normalized to TRUE so images will be scaled to max when retrieved
        self._normalized = True

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, normalized):
        if normalized is not self.normalized:
            self.__getitem__.cache_clear()
        if normalized:
            self.normalize_channels_to_max()

    ############################
    # Show random data samples #
    ############################
    def show_random(self):
        if self.dist_transformed:
            _, ax = plt.subplots(5, 1, figsize=(10, 10))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                ax[ii].plot(self[index][0].T, label=self.mode)
                ax[ii].legend()
                ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                   labelbottom=False)
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
                        ax[ii, jj].tick_params(top=False, bottom=False, left=False, right=False,
                                               labelleft=False,
                                               labelbottom=False)
                        ax[ii, jj].set_title(f'Response: {self[index][1]}. \n Mode: {self.mode[jj]}',
                                             fontsize=10)
                else:
                    ax[ii].imshow(transform(self[index][0][jj]))
                    ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                       labelbottom=False)
                    ax[ii].set_title(f'Response: {lab}. \n Mode: {self.mode[jj]}', fontsize=10)
        plt.show()
