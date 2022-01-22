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
# Create a random split of CheXpert samples.
#

parser = argparse.ArgumentParser()
parser.add_argument('--csv', metavar='CSV', required=True, nargs='+')
parser.add_argument('--output-a', metavar='CSV', required=True)
parser.add_argument('--output-b', metavar='CSV', required=True)
parser.add_argument('--a-prop', metavar='PROPORTION', default=0.8, type=float)
parser.add_argument('--seed', metavar='SEED', default=123, type=int)
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
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for line in tqdm.tqdm(reader, desc='preload'):
            # get scan properties
            patient_id, study_id = line['Path'].split('/')[2:4]
            view_number, view = re.match('.+view([0-9]+)_(frontal|lateral)', line['Path']).groups()
            view_number = int(view_number)
            subject_key = (patient_id, study_id)

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

np.random.seed(args.seed)
patients_list = sorted(list(patients))
np.random.shuffle(patients_list)

patients_a = set(patients_list[:int(len(patients_list) * args.a_prop)])
patients_b = set(patients_list[int(len(patients_list) * args.a_prop):])
print('Subset A', len(patients_a))
print('Subset B', len(patients_b))

with open(args.output_a, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for subject in paired_subjects.values():
        if subject['patient_id'] in patients_a:
            writer.writerow(subject['frontal']['line'])
            writer.writerow(subject['lateral']['line'])

with open(args.output_b, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for subject in paired_subjects.values():
        if subject['patient_id'] in patients_b:
            writer.writerow(subject['frontal']['line'])
            writer.writerow(subject['lateral']['line'])
