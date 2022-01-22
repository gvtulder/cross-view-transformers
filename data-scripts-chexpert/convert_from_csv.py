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
import csv
import os.path
import re
import imageio
import traceback
import tqdm
import numpy as np
import h5py

#
# Load CheXpert images and create an HDF5 file.
#

parser = argparse.ArgumentParser()
parser.add_argument('--csv', metavar='CSV', required=True, nargs='+')
parser.add_argument('--basedir', metavar='DIR', required=True)
parser.add_argument('--save-hdf5', metavar='HDF5',
                    help='save scans to hdf5')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


if args.verbose:
    def print_verbose(*args, **kwargs):
        print(*args, **kwargs)
else:
    def print_verbose(*args, **kwargs):
        pass

print_verbose(args)


subjects = {}

for filename in args.csv:
    with open(filename, 'r') as f:
        for line in tqdm.tqdm(csv.DictReader(f), desc='preload'):
            # get scan properties
            patient_id, study_id = line['Path'].split('/')[2:4]
            view_number, view = re.match('.+view([0-9]+)_(frontal|lateral)', line['Path']).groups()
            view_number = int(view_number)
            subject_key = (patient_id, study_id)

            line['patient_id'] = patient_id
            line['study_id'] = study_id
            line['view_number'] = view_number
            line['view'] = view

            if subject_key not in subjects:
                subjects[subject_key] = {'patient_id': patient_id, 'study_id': study_id}

            # take the last image for each view
            if view not in subjects[subject_key] or subjects[subject_key][view]['number'] < view_number:
                subjects[subject_key][view] = {'number': int(view_number), 'line': line}

# find subjects with frontal and lateral views
paired_subjects = {k: v for k, v in subjects.items() if 'frontal' in v and 'lateral' in v}
patients = set(v['patient_id'] for v in paired_subjects.values())

print('Total subjects:', len(subjects))
print('Complete subjects:', len(paired_subjects))
print('Complete patients:', len(patients))


LABELS = ('No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
          'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
          'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')
SCORE_MAP = {'': 99, '-1.0': -1, '0.0': 0, '1.0': 1}


if args.save_hdf5:
    h5out = h5py.File(args.save_hdf5, 'w')


for subject_idx, subject in enumerate(tqdm.tqdm(paired_subjects.values())):
    print_verbose('Patient %s, study %s' % (subject['patient_id'], subject['study_id']))

    if args.save_hdf5:
        scan_group = h5out.create_group('scans/%d' % subject_idx)

    scores = {}

    for view in ('frontal', 'lateral'):
        print_verbose('- %s' % view)
        img = imageio.imread(os.path.join(args.basedir, subject[view]['line']['Path']))
        print_verbose('  Image shape: ', img.shape)

        # normalize image to zero-mean, unit std
        img = img.astype(float)
        img_mean = img.mean()
        img_std = img.std()
        print_verbose('  Image mean, std:', img_mean, img_std)
        img -= img_mean
        img /= img_std

        # crop or pad image to 390 x 390
        if img.shape[0] > 390:
            crop_y = img.shape[0] - 390
            img = img[(crop_y // 2):(crop_y // 2 + 390), :]
        if img.shape[1] > 390:
            crop_x = img.shape[1] - 390
            img = img[:, (crop_x // 2):(crop_x // 2 + 390)]
        pad_y = 390 - img.shape[0]
        pad_x = 390 - img.shape[1]
        img = np.pad(img, ((pad_y // 2, pad_y - (pad_y // 2)),
                           (pad_x // 2, pad_x - (pad_x // 2))))
        print_verbose('  Sample shape:', img.shape)

        if args.save_hdf5:
            view_group = scan_group.create_group(view)
            view_group.create_dataset('image', data=img.astype('float32'), compression='gzip')
            view_group.attrs['path'] = subject[view]['line']['Path']
            view_group.attrs['img_mean'] = img_mean
            view_group.attrs['img_std'] = img_std

        for label in LABELS:
            view_score = subject[view]['line'][label]
            view_score = SCORE_MAP[view_score]
            assert label not in scores or scores[label] == view_score
            scores[label] = view_score

    if args.save_hdf5:
        scan_group.attrs['patient_id'] = subject['study_id']
        for label in LABELS:
            scan_group.attrs['score %s' % label] = scores[label]

    print_verbose()

