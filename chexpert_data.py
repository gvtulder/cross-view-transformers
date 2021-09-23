# Cross-view transformers for multi-view analysis of unregistered medical images
# Copyright (C) 2021 Gijs van Tulder / Radboud University, the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
import numpy as np
import skimage.transform
import elasticdeform
import torch
import torch.utils
import h5py

datasets = {}
def register_dataset(cls):
    datasets[cls.__name__] = cls
    return cls


# data loader for CheXpert data with two views
@register_dataset
class CheXpertDataset(torch.utils.data.Dataset):
    TASKS = ('No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
             'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
             'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')
    # in the original CSV: -1 uncertain
    #                       0 negative
    #                       1 positive
    #                      99 unknown
    LABEL_MAP = {99: 99, -1: 0, 0: 1, 1: 2}

    def __init__(self, datafile, augment=False, dtype=float, dtype_x=None, dtype_y=None,
                 views=['frontal', 'lateral']):
        super().__init__()
        self.datafile = datafile
        self.augment = augment
        self.dtype_x = dtype_x or dtype
        self.dtype_y = dtype_y or dtype
        self.views = views

        ds = h5py.File(self.datafile, 'r')
        self.scan_ids = list(ds['scans'])
        self.num_scans = len(self.scan_ids)

    def __getitem__(self, i):
        ds = h5py.File(self.datafile, 'r')
        scan = ds['scans'][self.scan_ids[i]]

        # load images for all views
        x_i = [scan[view]['image'][:] for view in self.views]

        # collect targets for all tasks
        # label uncertain / negative / positive for each task, or missing
        #          0           1           2                        -1
        y_i = [self.LABEL_MAP[scan.attrs['score %s' % task]] for task in self.TASKS]

        if self.augment:
            # flipping should be done for all views together
            coflip = np.random.randint(4)
            # augment each view independently
            x_i = [self.augment_image(x_i_v, coflip=coflip) for x_i_v in x_i]

        # convert each view's image to torch
        x_i = [torch.tensor(x_i_v[None, :, :], dtype=self.dtype_x) for x_i_v in x_i]

        # convert label to torch
        y_i = torch.tensor(y_i, dtype=self.dtype_y)

        # concatenate views + label
        return x_i + [y_i]

    def augment_image(self, x_i, coflip=None):
        # augment the image for one view
        if self.augment:
            x_i = x_i.astype(float)

            if 'flip' in self.augment or 'coflip' in self.augment:
                # flip (using sample setting for coflip)
                t = coflip if 'coflip' in self.augment else np.random.randint(4)
                if t == 1:  # flip first dimension
                    x_i = x_i[::-1, :]
                elif t == 2:  # flip second dimension
                    x_i = x_i[:, ::-1]
                elif t == 3:  # flip both dimensions
                    x_i = x_i[::-1, ::-1]

            if 'elastic' in self.augment:
                # elastic deformations
                t = np.random.randint(2)
                if t == 1:
                    # choose a random zoom factor [0.9, 1.1]
                    zoom = np.random.uniform(0.9, 1.1)
                    # choose a random rotation of [-30, +30] degrees
                    rotate = np.random.uniform(-30, 30)
                    x_i = elasticdeform.deform_random_grid(x_i,
                            sigma=5, points=5, zoom=zoom, rotate=rotate)

            if 'crop20' in self.augment:
                # random crop 20 pixels on each side
                offset_x = np.random.randint(40)
                offset_y = np.random.randint(40)
                x_i = x_i[offset_y:-(40 - offset_y), offset_x:-(40 - offset_x)]

            if 'gaussiannoise' in self.augment:
                x_i = np.random.normal(x_i, 0.01)

            # pytorch does not like negative strides
            x_i = np.ascontiguousarray(x_i)

        return x_i

    def __len__(self):
        return self.num_scans
