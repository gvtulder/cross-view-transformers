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
import tqdm
import h5py
from collections import defaultdict

#
# Merge CC and MLO views for each subject.
#

parser = argparse.ArgumentParser()
parser.add_argument('--source-cc', metavar='HDF5', required=True,
                    help='source file with CC scans')
parser.add_argument('--source-mlo', metavar='HDF5', required=True,
                    help='source file with MLO scans')
parser.add_argument('--save-hdf5', metavar='HDF5', required=True,
                    help='save scans to hdf5')
args = parser.parse_args()


# inputs
h5in = {
    'cc': h5py.File(args.source_cc, 'r'),
    'mlo': h5py.File(args.source_mlo, 'r')
}
views = ('cc', 'mlo')

# output
h5out = h5py.File(args.save_hdf5, 'w')

# match subjects
subjects = defaultdict(dict)
for view in h5in:
    for scan in tqdm.tqdm(h5in[view]['scans'].values(), desc='index %s subjects' % view):
        # match subject and side, but not view
        normalized_subject_id = '_'.join([scan.attrs['patient_id'], scan.attrs['side']])
        subjects[normalized_subject_id][view] = scan

# print statistics
print()
print('Total subjects      %4d' % len(subjects))
for view in views:
    print('- Subjects in %-3s   %4d' % (view, len([s for s in subjects.values() if view in s])))
print('------------------------')

# complete_subjects
complete_subjects = [s for s in subjects if len(subjects[s]) == len(views)]
print('Complete subjects   %4d' % len(complete_subjects))
print()

# save subjects with scans from both views
subject_idx = 0
for subject_id in tqdm.tqdm(complete_subjects, desc='Merging'):
    subject_views = subjects[subject_id]
    if len(views) == len(subject_views):
        # add complete subject
        common_attrs = {}
        for view in views:
            # copy image and segmentation for this view
            h5out.create_dataset('scans/%d/%s/image' % (subject_idx, view),
                                 data=subject_views[view]['image'], compression='gzip')
            h5out.create_dataset('scans/%d/%s/segmentation' % (subject_idx, view),
                                 data=subject_views[view]['segmentation'], compression='gzip')

            # copy all attributes for this view
            group = h5out['scans/%d/%s' % (subject_idx, view)]
            for key, value in subject_views[view].attrs.items():
                group.attrs[key] = value

            # keep common attributes
            for key in ('patient_id', 'side'):
                value = subject_views[view].attrs[key]
                if key not in common_attrs:
                    common_attrs[key] = value 
                elif value != common_attrs[key]:
                    print('Views have different %s!' % patient_id)
                    print('Current view:', subject_views[view].attrs)
                    raise Exception()

            # select worst assessment label
            assessment_label = subject_views[view].attrs['assessment_label']
            if 'assessment_label' not in common_attrs or assessment_label == 'malignant':
                common_attrs['assessment_label'] = assessment_label 


        # store overall attributes for this subject
        group = h5out['scans/%d' % subject_idx]
        for key in ('patient_id', 'side', 'assessment_label'):
            group.attrs[key] = common_attrs[key]

        subject_idx += 1
