import warnings

import torchvision.transforms.v2 as t
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

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
        - normalized_to_max (bool): Returns whether image stack is normalized_to_max
        - show_random (callable): Shows 5 random samples from dataset
        - device (str or torch.device): Device type or device ID
        - to (callable): Move any currently cached items to DEVICE from input argument in call and return all future
            items on DEVICE.
    """

    # region Main Dataset Methods (init, len, getitem)
    def __init__(self, root, mode, xl_file=None, label=None, mask_on=True, transforms=None,
                 use_atlas=False, use_patches=True, patch_size=(512, 512), use_cache=True,
                 device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.device = device
        self.label = label
        self._use_atlas = use_atlas
        self.mode = mode if type(mode) is list else [mode]
        self.mask_on = False if self._use_atlas else mask_on
        self.use_cache = use_cache
        self.use_patches = use_patches if self._use_atlas else False
        self._patch_size = patch_size

        # Set attribute and property defaults
        self.saturate = False
        self.augmented = False
        self.filter_bad_data = False
        self.dist_transformed = False
        self.psuedo_rgb = False
        self._name = 'nsclc_'
        self._shape = None
        self._normalized_to_max = False
        self._min_max_scalars = None
        self._normalized_to_preset = False
        self._normalized = False
        self._nbins = 25

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

        # Define a mode dict that matches appropriate load functions and filename regex to mode
        self.mode_dict = {'mask': [load_fn['tiff'], rf'.*?(\\|/)Redox(\\|/)ROI_mask\.(tiff|TIFF)'],
                          'orr': [load_fn['tiff'], rf'.*?(\\|/)Redox(\\|/)RawRedoxMap\.(tiff|TIFF)'],
                          'g': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_phasor_G\.(asc|ASC)'],
                          's': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_phasor_S\.(asc|ASC)'],
                          'photons': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_photons\.(asc|ASC)'],
                          'tau1': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_t1\.(asc|ASC)'],
                          'tau2': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_t2\.(asc|ASC)'],
                          'alpha1': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_a1\.(asc|ASC)'],
                          'alpha2': [load_fn['asc'], rf'.*?(\\|/)FLIM(\\|/).*?_a2\.(asc|ASC)']}
        # Compile regexes
        self.mode_dict = {key: [item[0], re.compile(item[1])] for key, item in self.mode_dict.items()}

        # Set max value presets for normalization and/or saturation
        '''
        These preset values are based on the mean and standard deviation of the dataset for each mode. As a general 
        rule, when data is normally distributed, 99.7% of samples fall within 3 standard deviations of the mean. We will
        use that then as the cutoffs for our presets. Theses were performed for masked data across all modes using the 
        following code:
        
            import numpy as np
            x = torch.tensor([])
            for i in range(len(data)):
                x = torch.cat((x, data[i][0].unsqueeze(0)), dim=0)
            torch.nanmean(torch.nanmean(torch.nanmean(x, dim=0), dim=1), dim=1)
            np.nanstd(np.nanstd(np.nanstd(x, axis=0), axis=1), axis=1)
        
        where 'data' is an instance of this class with all modes (note, not 'all' set as the mode, because this doesn't 
        actually give all modes)
        
        The results for each mode are as follows:
        Mode                    Mean                StDev
        ORR                 3.5064e-01          2.9563567e-02    
        G                   3.5540e-01          1.1096316e-02
        S                   3.7959e-01          3.8442286e-03
        Photons             1.9098e+02          1.2940987e+01
        Tau1                651.6812            45.86681         
        Tau2                3834.9180           193.67015
        Alpha1              121.3796            9.293237
        Alpha2              83.6695             5.76365
        TauMean             2.0095e+03          9.8343758e+01
        BoundFraction       4.2615e-01          1.6073210e-02 
        
        For reference, that is included as a dict below, which will be used to calculate the ranges for all modes.
        '''
        mean_std_mode_dict = {'ORR': [3.5064e-01, 2.9563567e-02],
                              'G': [3.5540e-01, 1.1096316e-02],
                              'S': [3.7959e-01, 3.8442286e-03],
                              'Photons': [1.9098e+02, 1.2940987e+01],
                              'Tau1': [651.6812, 45.86681],
                              'Tau2': [3834.9180, 193.67015],
                              'Alpha1': [121.3796, 9.293237],
                              'Alpha2': [83.6695, 5.76365],
                              'TauMean': [2.0095e+03, 9.8343758e+01],
                              'BoundFraction': [4.2615e-01, .6073210e-02]}

        self._preset_values = {}
        for key, (m, s) in mean_std_mode_dict.items():
            self._preset_values[key.lower()] = [m - 3 * s, m + 3 * s]

        # Find and load features spreadsheet (or load directly if path provided)
        if xl_file is None:
            xl_file = glob.glob(self.root + os.sep + '*.xlsx')
            if not xl_file:
                raise Exception(
                    'Features file not found,'
                    ' input path manually at dataset initialization using xl_file=<path_to_file>.')
            self.xl_file = xl_file[0]
        self.features = pd.read_excel(self.xl_file)
        self.patient_count = len(self.features)
        """
        - atlases_by_sample and fovs_by_slide are lists of lists. The inner lists correspond to the images within a 
        given sample or slide and the outer lists match the index of the slide to the features. In other words, the 
        index of a given slide in the features file will also match a list of all images from that slide in the list of 
        lists.
        - all_atlases and all_fovs maintain the order, but un-nest the lists so they can easily be indexed into 
        to actually use the paths to get items.  
        - atlas_mode_dict and fov_mode_dict are lists of dicts. Each dict matches the index of the all_... lists, but 
        also includes functions and paths for individual modes. They are of the form:
            ..._mode_dict[<image_index_from_all_list>] = {<mode name>: [<specific load function for mode's file type>,
                                                                        <path for specific mode of indexed image and>]
                                                           ...}
           In the case of atlases, this mode dict is trivial because there is only one load function (since only 'orr' 
           is available), but it is still produced to easily match loading at __getitem__ for both data types.
           
           Whichever data type will be used is added as simply img_... variables to easily pass to __getitem__ 
           regardless of datatype. After __init__ this should make the behavior the same (with the exception of cropping
           and __len__ for atlases).      
        """
        # Prepare a list of images from data dir matched to slide names from the features excel file
        if self._use_atlas:
            # Track number of individual 512x512 patches available within each atlas
            self.atlas_patch_dims = []

            # Track running total of patches through each index for sub-indexing later
            total_patches_through_index = 0
            self.atlas_sub_index_map = [0]

            # Nested list of atlas locations nested by sample (for label indexing)
            self.atlases_by_sample = []

            # A "mode_dict" that looks the same as the fov_mode_dict (though this one is trivial)
            self.atlas_mode_dict = []
            for subject in self.features['Sample ID']:
                sample_dir = os.path.join(self.root, 'Atlas_Images', subject)
                self.atlases_by_sample.append([])
                for trunk, dirs, files in os.walk(sample_dir):
                    for f in files:
                        if 'rawredoxmap.tiff' == f.lower():
                            im_path = os.path.join(trunk, f)
                            self.atlases_by_sample[-1].append(trunk)
                            self.atlas_mode_dict.append({'orr': [load_tiff, im_path]})
                            with Image.open(im_path) as im:
                                width, height = im.size
                                rm_width, rm_height = width % self._patch_size[1], height % self._patch_size[0]
                                width, height = width - rm_width, height - rm_height  # "crop" to be a perfect fit
                                self.atlas_patch_dims.append(
                                    (height / self._patch_size[0], width / self._patch_size[1]))
                                total_patches_through_index += np.prod(self.atlas_patch_dims[-1])
                                self.atlas_sub_index_map.append(total_patches_through_index)

            # Un-nest list (still ordered, but now easily indexable)
            self.all_atlases = [atlas for sample_atlases in self.atlases_by_sample for atlas in sample_atlases]
        else:
            self.fovs_by_slide = [glob.glob(self.root + os.sep + slide + '*') for slide in
                                  self.features['Slide Name']]  # This gives a list of lists of fovs sorted by slide
            self.all_fovs = [fov for slide_fovs in self.fovs_by_slide for fov in
                             slide_fovs]  # This will give a list of all fovs (still ordered, but now not nested,
            # making it simple for indexing in __getitem__)

            # Make an indexed FOV-LUT dict list of loaders and files
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
                    if sum(path_str is not None for path_str in matched) == 1:
                        for match in matched:
                            if match:
                                self.fov_mode_dict[index][mode] = [load_fn, match.string]
                    # Else, add <None> for later removal and move on to next FOV
                    else:
                        self.fov_mode_dict[index][mode] = [fov, None]

                # Add derived modes
                self.fov_mode_dict[index]['boundfraction'] = [load_bound_fraction, [self.fov_mode_dict[index]['alpha1'],
                                                                                    self.fov_mode_dict[index]['alpha2']
                                                                                    ]]
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
        if self._use_atlas:
            if self.use_patches:
                atlas_len = int(self.atlas_sub_index_map[-1])
            else:
                atlas_len = len(self.all_atlases)
            if self.augmented:
                return 5 * atlas_len
            else:
                return atlas_len
        else:
            if self.augmented:
                return 5 * len(self.all_fovs)
            else:
                return len(self.all_fovs)

    def __getitem__(self, index):
        # Parse the index
        # Get image path from index
        if self.augmented:
            base_index = int(index // 5)  # This will give us the index for the base img
            sub_index = index % 5  # This gives the index of the crop
        else:
            base_index = index
        if self._use_atlas:
            # Find where the index is <= than the number of patches
            if self.use_patches:
                path_index = 0
                while self.atlas_sub_index_map[path_index + 1] <= base_index:
                    path_index += 1
                patch_index = int(base_index - self.atlas_sub_index_map[path_index])
            else:
                path_index = base_index
            img_path = self.all_atlases[path_index]
            load_dict = self.atlas_mode_dict
            lists_of_paths = self.atlases_by_sample
        else:
            path_index = base_index
            img_path = self.all_fovs[path_index]
            load_dict = self.fov_mode_dict
            lists_of_paths = self.fovs_by_slide
        slide_idx = [img_path in paths for paths in lists_of_paths].index(True)

        # Check if indexed sample is in cache (by checking for index in index_cache)...
        # if it is, pull it from the cache;
        # Get base image and label from cache
        if self.index_cache is not None and path_index in self.index_cache:
            x = self.shared_x[path_index].clone()
            y = self.shared_y[path_index]
        # Load base image and label into cache
        else:
            # load the sample, and cache the sample (if cache is open)
            # region Load Data and Label

            # Load modes using load functions
            for ii, mode in enumerate(self.mode):
                # Pre-allocate on first pass
                mode_load = load_dict[path_index][mode][0](load_dict[path_index][mode]).to(self.device)
                if ii == 0:
                    self.image_dims = (self.stack_height,) + tuple(mode_load.size()[1:])
                    x = torch.empty(self.image_dims, dtype=torch.float32, device=self.device)
                x[ii] = mode_load

            # Get data label
            # Get index of nested list that contains the image path based on what slide the FOV is from. This index will
            # coincide with the index of the features file to get the label of the sample/slide the image is from.
            match self.label:
                case 'Response':
                    y = torch.tensor(1 if self.features['Status (NR/R)'].iloc[slide_idx] == 'R' else 0,
                                     dtype=torch.float32, device=self.device)
                case 'Metastases':
                    y = torch.tensor(1 if self.features['Status (Mets/NM)'].iloc[slide_idx] == 'NM' else 0,
                                     dtype=torch.float32, device=self.device)
                case 'Mask':
                    # Load mask (if on or label)
                    fov_mask = load_dict[path_index]['mask'][0](load_dict[path_index]['mask']).to(self.device)
                    fov_mask[fov_mask == 0] = float('nan')
                    y = fov_mask
                case None:
                    y = torch.tensor(-999999,  # Placeholder for NaN label
                                     dtype=torch.float32, device=self.device)
                case _:
                    raise Exception(
                        'An unrecognized label is in use. Update label attribute of dataset and try again.')

            # Add the loaded image data to the cache (open and add, if it's not open)
            if self.index_cache is None and self.use_cache:
                self._open_cache(x, y)
            if self.use_cache:
                self.shared_x[path_index] = x.clone()
                self.shared_y[path_index] = y.clone()
                self.index_cache[path_index] = path_index

        # Load mask (if on)
        if self.mask_on:
            fov_mask = load_dict[path_index]['mask'][0](load_dict[path_index]['mask']).to(self.device).squeeze()
            x[:, fov_mask == 0] = float('nan')

        # Perform all data augmentations, transformations, etc. on base image
        # patch the atlas into individual images
        if self.use_patches:
            r = int(self.atlas_patch_dims[path_index][0] * self._patch_size[0])
            c = int(self.atlas_patch_dims[path_index][1] * self._patch_size[1])
            x = x[:, :r, :c]
            x = x.unfold(1, self._patch_size[0], self._patch_size[0])
            x = x.unfold(2, self._patch_size[1], self._patch_size[1])
            x = x.reshape(self.stack_height, -1, self._patch_size[0], self._patch_size[1])
            x = x[:, patch_index, :, :]

        # Crop and sub index if necessary
        if self.augmented:
            cropper = t.FiveCrop((int(x.shape[1] / 2), int(x.shape[2] / 2)))
            x = cropper(x)
            x = x[sub_index]

        # Cutoff at saturation thresholds
        if self.saturate:
            for ch, mode in enumerate(self.mode):
                x[x[ch > self._preset_values[mode]]] = self._preset_values[mode]

        # Scale by the scalars from normalization method
        if self._normalized:
            for ch in range(len(self.mode)):
                x[ch] = (x[ch] - self.scalars[ch, 0]) / (self.scalars[ch, 1] - self.scalars[ch, 0])

        # Dynamically and recursively filter out bad data (namely, images with little or no signal) while maintain the
        # same patient index
        if self.filter_bad_data and self.is_bad_data(x):
            print(f'crap {index}')
            pt_indices = self.get_patient_subset(slide_idx)
            new_idx = pt_indices[(pt_indices.index(index) + 1) % len(pt_indices)]
            return self.__getitem__(new_idx)

        # Unsqueeze so "color" is dim 1 and expand to look like an RGB image
        # New image dims: (M, C, H, W), where M is the mode, C is the psuedo-color channel, H and W are height and width
        if self.psuedo_rgb:
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)

        # Apply distribution transform, if called
        if self.dist_transformed:
            x_dist = torch.empty((self.stack_height,) + (self._nbins,), dtype=torch.float32, device=self.device)
            for ch, mode in enumerate(x):
                x_dist[ch], _ = torch.histogram(mode.cpu(), bins=self._nbins, range=[0, 1], density=True)
            x = x_dist

        # Apply transforms that were input (if any)
        x = self.transforms(x) if self.transforms is not None else x

        return x, y
        # endregion

    def _open_cache(self, x, y):
        # Setup shared memory arrays (i.e. caches that are compatible with multiple workers)
        # negative initialization ensure no overlap with actual cached indices
        cache_len = len(self.all_atlases) if self._use_atlas else len(self.all_fovs)
        index_cache_base = mp.Array(ctypes.c_int, cache_len * [-1])
        shared_x_base = cache_len * [mp.Array(ctypes.c_float, 0)]

        # Label-size determines cache size, so if no label is set, we will fill cache with -999999 at __getitem__
        match self.label:
            case 'Response' | 'Metastases' | None:
                shared_y_base = mp.Array(ctypes.c_float, cache_len * [-1])
                y_shape = ()
            case 'Mask':
                shared_y_base = mp.Array(ctypes.c_float, int(cache_len * np.prod(y.shape)))
                y_shape = tuple(y.shape)
            case _:
                raise Exception('An unrecognized label is in use that is blocking the cache from initializing. '
                                'Update label attribute of dataset and try again.')

        # Convert all arrays to desired data structure
        self.index_cache = convert_mp_to_torch(index_cache_base, (cache_len,), device=self.device)
        self.shared_x = [convert_mp_to_torch(x_base, 0, device=self.device) for x_base in shared_x_base]
        self.shared_y = convert_mp_to_torch(shared_y_base, (cache_len,) + y_shape, device=self.device)
        print('Cache opened.')

    def to(self, device):
        # Move caches to device
        if self.index_cache is not None:
            self.index_cache = self.index_cache.to(device)
            for i, idx in enumerate(self.index_cache):
                self.index_cache[i] = idx.to(device)
            for i, x_cache in enumerate(self.shared_x):
                self.shared_x[i] = x_cache.to(device)
            self.shared_y = self.shared_y.to(device)
            for i, y in enumerate(self.shared_y):
                self.shared_y[i] = y.to(device)

        # Move any self-held tensors to device for ops compatibility
        self.scalars = self.scalars.to(device) if self.scalars is not None else None

        # Update device for future items
        self.device = device

    def get_patient_subset(self, pt_index):
        # Get actual index from input index (to handle negatives)
        pt_index = list(range(len(self.features)))[pt_index]
        pt_id = self.features.at[pt_index, 'Sample ID'] if self._use_atlas else self.features.at[pt_index, 'Slide Name']
        if self._use_atlas:
            indices = [i for i, path_str in enumerate(self.all_atlases) if pt_id in path_str]
        else:
            indices = [i for i, path_str in enumerate(self.all_fovs) if pt_id in path_str]

        # If using atlas patches, we have to determine how many patches come before this patient and add the number from
        # the patient from there
        if self.use_patches:
            im_before_current_pt = int(self.atlas_sub_index_map[indices[0]])
            number_of_ims_for_pt = int(sum([np.prod(self.atlas_patch_dims[i]) for i in indices]))
            indices = list(range(im_before_current_pt, im_before_current_pt + number_of_ims_for_pt))

        # If using augmenting, each base image results in 5 sequential daughter images
        if self.augmented:
            indices = [5 * idx + i for idx in indices for i in range(5)]

        return indices

    def get_patient_label(self, pt_index):
        match self.label:
            case 'Response':
                y = torch.tensor(1 if self.features.at[pt_index, 'Status (NR/R)'] == 'R' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Metastases':
                y = torch.tensor(1 if self.features.at[pt_index, 'Status (Mets/NM)'] == 'NM' else 0,
                                 dtype=torch.float32, device=self.device)
            case 'Mask':
                # Load mask (if on or label)
                pass
            case None:
                y = torch.tensor(-999999,  # Placeholder for NaN label
                                 dtype=torch.float32, device=self.device)
            case _:
                raise Exception(
                    'An unrecognized label is in use. Update label attribute of dataset and try again.')
        return y

    def is_bad_data(self, x):
        ## TODO: Write bad data check and add check for atlases to removed and move to next index of the sample at get item ##
        return (torch.sum(x <= 0.1).item() + torch.sum(x >= 0.9).item()) > (0.60 * np.prod(x.shape))

    # endregion

    # region Properties
    # Name, shape, label, mode, mask
    # Use of property (instead of simple attribute) to define ensures automatic updates with latest data setup and/or
    # appropriate clearing of the cache

    # Name
    @property
    def name(self):
        self._name = (f"nsclc_{self.label}_{'+'.join(self.mode)}"
                      f'{"_Histogram" if self.dist_transformed else ""}'
                      f'{"_Augmented" if self.augmented else ""}'
                      f'{"_Normalized" if self._normalized else ""}'
                      f'{"_Masked" if self.mask_on else ""}')
        return self._name

    # Shape
    @property
    def shape(self):
        temp_filter = self.filter_bad_data
        self.filter_bad_data = False
        if self._use_atlas and not self.use_patches:
            warnings.warn('`shape` is ambiguous when using non-patched atlases.')
        self._shape = self[0][0].shape
        self.filter_bad_data = temp_filter
        return self._shape

    # Label
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        assert label is not None, 'Label must be provided.'
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
            temp_filter = self.filter_bad_data
            self._open_cache(*self[0])
            self.filter_bad_data = temp_filter
        self._label = label

    # Modes
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # Set default/shortcut behavior
        mode = [mode] if type(mode) is not list else mode
        if mode is None:
            mode = ['orr'] if self._use_atlas else ['all']
        if mode == ['all']:
            mode = ['orr', 'g', 's', 'photons', 'taumean', 'boundfraction']
        if self._use_atlas and mode[:] == 'orr':
            warnings.warn(f"{mode} is incompatible with atlases. Mode is being reset to 'orr'.")
            mode = ['orr']
        # If this is not the __init__ run
        if hasattr(self, '_mode') and mode != self.mode:
            # A mode update is essentially a new dataset, so we will re-init it, but we want to make all the other
            # aspects match, so we check and store them first. Once the dataset is reinitialized, we can reset the
            # other properties.
            temp_aug = self.augmented
            temp_norm_max = self.normalized_to_max
            temp_norm_preset = self.normalized_to_preset
            temp_dist = self.dist_transformed

            # Re-init with new modes
            self.__init__(self.root, mode, xl_file=self.xl_file, label=self.label, mask_on=self.mask_on,
                          transforms=self.transforms, device=self.device)

            # Reset properties
            self.augmented = temp_aug
            self.normalized_to_max = temp_norm_max
            self.normalized_to_preset = temp_norm_preset
            self.dist_transformed = temp_dist
        # If this is the __init__ run
        else:
            self._mode = mode

    # endregion

    # region Transforms and Augmentations
    def dist_transform(self, nbins=25):
        # If it is already transformed to the same bin number, leave it alone
        if self.dist_transformed and nbins == self._nbins:
            pass
        # If something is changed, reset and update
        else:
            # self.reset_cache()
            self._nbins = nbins
            if not self._normalized:
                print(
                    'Normalization to presets is automatically applied for the distribution transform.\n     '
                    'This can be manually overwritten by setting the NORMALIZED attribute to False after transforming. '
                    "To use max normalization, use normalize_channels('max') before transforming.")
                self.normalize_channels('presets')
        self.dist_transformed = True

    def augment(self):
        self.augmented = True

    def transform_to_psuedo_rgb(self):
        self.psuedo_rgb = True

    def cutoff_saturation(self):
        self.saturate = True

    # region Normalization
    @property
    def normalized_to_max(self):
        return self._normalized_to_max

    @normalized_to_max.setter
    def normalized_to_max(self, normalized_to_max):
        self._normalized_to_max = normalized_to_max
        self.normalized_to_preset = False if not self._normalized else not normalized_to_max
        self._normalized = (normalized_to_max or self.normalized_to_preset)

    @property
    def normalized_to_preset(self):
        return self._normalized_to_preset

    @normalized_to_preset.setter
    def normalized_to_preset(self, normalized_to_preset):
        self._normalized_to_preset = normalized_to_preset
        self.normalized_to_max = False if not self._normalized else not normalized_to_preset
        self._normalized = (normalized_to_preset or self.normalized_to_max)

    def normalize_channels(self, method='minmax'):
        match method.lower():
            case 'minmax' | 'max':
                if self._min_max_scalars is None:
                    # Find the max for each mode across the entire dataset. This is mildly time-consuming, so we only do
                    # it once, then store the scalar and mark the set as normalized_to_max. In order to make
                    # distributions consistent, this step will be required for dist transforms, so it will be checked
                    # before performing the transform

                    # Temporarily turn psuedo_rgb off (if on) so we can save 2/3 memory and not have to worry about dim
                    # shifts for both cases. Temporarily turn off patching so we can save all the additional ops and just
                    # load straight from once
                    temp_psuedo = self.psuedo_rgb
                    self.psuedo_rgb = False
                    temp_patch = self.use_patches
                    self.use_patches = False

                    # Preallocate an array. Each row is an individual image, each column is mode
                    maxes = torch.zeros(len(self), self.stack_height, dtype=torch.float32, device=self.device)
                    mins = torch.zeros(len(self), self.stack_height, dtype=torch.float32, device=self.device)
                    for ii, (stack, _) in enumerate(self):
                        # Does the same as np.nanmax(stack, dim=(1,2)) but keeps the tensor on the GPU
                        maxes[ii] = torch.max(torch.max(torch.nan_to_num(stack, nan=-100000), 1).values, 1).values
                        mins[ii] = torch.min(torch.min(torch.nan_to_num(stack, nan=100000), 1).values, 1).values
                    self._min_max_scalars = torch.stack((torch.max(mins, 0).values, torch.max(mins, 0).values), dim=-1)

                    # Reset psuedo_rgb and patching
                    self.psuedo_rgb = temp_psuedo
                    self.use_patches = temp_patch
                self.scalars = self._min_max_scalars
                # Set normalized_to_max to TRUE so images will be scaled to max when retrieved
                self._normalized_to_max = True
            case 'preset':
                self.scalars = torch.tensor([self._preset_values[mode] for mode in self.mode],
                                            dtype=torch.float32, device=self.device)
                self._normalized_to_preset = True

    # endregion
    # endregion

    # region Show random data samples
    def show_random(self, stack_as_rgb=False):
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
            if stack_as_rgb:
                _, ax = plt.subplots(1, 5, figsize=(25, 25))
            else:
                _, ax = plt.subplots(5, self.stack_height, figsize=(25, 25))
            for ii in range(5):
                index = np.random.randint(0, len(self))
                img = self[index][0]
                img[torch.isnan(img)] = 0
                lab = self[index][1]
                if self.stack_height > 1 and not stack_as_rgb:
                    for jj in range(self.stack_height):
                        ax[ii, jj].imshow(transform(img[jj]))
                        ax[ii, jj].tick_params(top=False, bottom=False, left=False, right=False,
                                               labelleft=False,
                                               labelbottom=False)
                        ax[ii, jj].set_title(f'{self.label}: {self[index][1]}. \n Mode: {self.mode[jj]}',
                                             fontsize=10)
                else:
                    ax[ii].imshow(transform(img))
                    ax[ii].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                                       labelbottom=False)
                    ax[ii].set_title(f'{self.label}: {lab}. \n Mode: {self.mode[:]}', fontsize=10)
        plt.show()

    # endregion
