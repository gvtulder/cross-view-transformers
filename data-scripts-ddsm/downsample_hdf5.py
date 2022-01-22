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
 
import argparse
import h5py
import numpy as np
import skimage.transform
import tqdm

# resample images to the given scale

parser = argparse.ArgumentParser()
parser.add_argument('--input', metavar='HDF5', required=True)
parser.add_argument('--output', metavar='HDF5', required=True)
parser.add_argument('--scale', metavar='SCALE', type=float, default=0.5)
args = parser.parse_args()


def copy_attrs(src, tgt):
    for k, v in src.attrs.items():
        tgt.attrs[k] = v


h5in = h5py.File(args.input, 'r')
h5out = h5py.File(args.output, 'w')

for scan_id, scan_in in tqdm.tqdm(h5in['scans'].items()):
    names = []
    scan_in.visit(lambda name: names.append(name))

    # copy attributes
    scan_out = h5out.create_group('scans/%s' % scan_id)
    copy_attrs(scan_in, scan_out)

    for name in names:
        obj_in = scan_in[name]
        if isinstance(obj_in, h5py.Dataset):
            img = obj_in[:]
            img = skimage.transform.rescale(img.astype('float64'), args.scale,
                                            anti_aliasing=True, order=3)
            if obj_in.dtype == np.bool:
                # threshold to include partial masks
                img = img > 0
            else:
                img = img.astype(obj_in.dtype)
            obj_out = scan_out.create_dataset(name, compression='gzip', data=img)
        elif isinstance(obj_in, h5py.Group):
            obj_out = scan_out.create_group(name)
        else:
            raise Exception('Unknown object type:', obj)
        copy_attrs(obj_in, obj_out)
