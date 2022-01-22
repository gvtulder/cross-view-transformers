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
import numpy as np
import tqdm
import h5py

#
# Create random subsets.
#

parser = argparse.ArgumentParser()
parser.add_argument('--sources', metavar='HDF5', required=True, nargs='+',
                    help='source files')
parser.add_argument('--save-hdf5-base', metavar='HDF5', required=True,
                    help='save scans to hdf5')
parser.add_argument('--subsets', metavar='N', default=5, type=int)
parser.add_argument('--seed', metavar='SEED', default=123, type=int)
args = parser.parse_args()

patient_labels = {}
patients_per_class = {'benign': [], 'malignant': []}
for source in args.sources:
    d = h5py.File(source, 'r')
    for scan_id in d['scans']:
        patient_id = d['scans'][scan_id].attrs['patient_id']
        label = d['scans'][scan_id].attrs['assessment_label']

        # assign each patient to only one label
        if patient_id in patient_labels:
            label = patient_labels[patient_id][0]
        else:
            patient_labels[patient_id] = [label, []]

        patient_labels[patient_id][1].append([source, scan_id])
        patients_per_class[label].append(patient_labels[patient_id])

print('%d benign patients' % len(patients_per_class['benign']))
print('%d malignant patients' % len(patients_per_class['malignant']))

np.random.seed(args.seed)
np.random.shuffle(patients_per_class['benign'])
np.random.shuffle(patients_per_class['malignant'])

patient_lists = [[] for i in range(args.subsets)]

subset = 0
for label in patients_per_class:
    while len(patients_per_class[label]) > 0:
        _, patient_scans = patients_per_class[label].pop()
        patient_lists[subset] += patient_scans
        subset = (subset + 1) % args.subsets

for i in range(args.subsets):
    print('Subset %d: %d patients' % (i, len(patient_lists[i])))
    h5out = h5py.File('%s-subset%s.h5' % (args.save_hdf5_base, i), 'w')
    for idx, (source, source_scan_id) in enumerate(tqdm.tqdm(patient_lists[i])):
        h5out.copy(h5py.File(source, 'r')['scans'][source_scan_id], '/scans/%d' % idx)

