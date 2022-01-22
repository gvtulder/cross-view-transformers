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
import traceback
import tqdm
import numpy as np
import h5py
import pydicom
import skimage.transform

import nyu_cropping

#
# Load CBIS-DDSM DICOM files and create an HDF5 file.
#

parser = argparse.ArgumentParser()
parser.add_argument('--csv', metavar='CSV', required=True, nargs='+')
parser.add_argument('--basedir', metavar='DIR', required=True)
parser.add_argument('--only-view', choices=('CC', 'MLO'))
parser.add_argument('--crop-size', metavar='PX', type=int, nargs=2,
                    help='size of crop (after rescaling)')
parser.add_argument('--crop-method', choices=('centroid', 'nyu'), default='centroid',
                    help='cropping method')
parser.add_argument('--rescale', metavar='F', type=float,
                    help='rescale factor')
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

ASSESSMENT_MAP = {'BENIGN_WITHOUT_CALLBACK': 'benign',
                  'BENIGN': 'benign', 'MALIGNANT': 'malignant'}
ASSESSMENTS = ('benign', 'malignant')
SIDE_MAP = {'LEFT': 'left', 'RIGHT': 'right'}
ABNORMALITY_MAP = {'calcification': 'calcification', 'mass': 'mass'}
VIEW_MAP = {'CC': 'cc', 'MLO': 'mlo'}

for filename in args.csv:
#   print_verbose(filename)
    with open(filename, 'r') as f:
        for line in tqdm.tqdm(csv.DictReader(f), desc='preload'):
            scan_key = (line['patient_id'], line['left or right breast'],
                        line['image view'], line['abnormality type'])
            side = SIDE_MAP[line['left or right breast']],
            view = VIEW_MAP[line['image view']],

            if args.only_view and args.only_view != line['image view']:
                continue

            image_file_path = os.path.join(args.basedir, line['image file path'].strip())
            crop_file_path = os.path.join(args.basedir, line['cropped image file path'].strip())
            mask_file_path = os.path.join(args.basedir, line['ROI mask file path'].strip())
            if not os.path.exists(image_file_path):
                print_verbose('File not found: %s' % image_file_path)
            elif not os.path.exists(crop_file_path):
                print_verbose('File not found: %s' % crop_file_path)
            elif not os.path.exists(mask_file_path):
                print_verbose('File not found: %s' % mask_file_path)
            else:
                with pydicom.filereader.dcmread(image_file_path, stop_before_pixels=True) as d:
                    img_size = (d['Rows'].value, d['Columns'].value)
                with pydicom.filereader.dcmread(crop_file_path, stop_before_pixels=True) as d:
                    crop_size = (d['Rows'].value, d['Columns'].value)
                with pydicom.filereader.dcmread(mask_file_path, stop_before_pixels=True) as d:
                    mask_size = (d['Rows'].value, d['Columns'].value)

                if crop_size[0] > mask_size[0]:
#                   print_verbose('  ' + image_file_path)
#                   print_verbose('    img:', img_size, img_size[0] / img_size[1])
#                   print_verbose('    crop:', crop_size, crop_size[0] / crop_size[1])
#                   print_verbose('    mask:', mask_size, mask_size[0] / mask_size[1])
#                   print_verbose('  SWAP mask and crop')
                    crop_file_path, mask_file_path = mask_file_path, crop_file_path
                    crop_size, mask_size = mask_size, crop_size

                if scan_key not in subjects:
                    subjects[scan_key] = {
                        'patient_id':       line['patient_id'],
                        'side':             SIDE_MAP[line['left or right breast']],
                        'view':             VIEW_MAP[line['image view']],
                        'density_score':    int(line['breast density'] if 'breast density' in line else line['breast_density']),
                        'assessment_score': int(line['assessment']),
                        'assessment_label': ASSESSMENT_MAP[line['pathology']],
                        'lesion_type':      ABNORMALITY_MAP[line['abnormality type']],
                        'image_file':       image_file_path,
                        'mask_file':        [],
                    }

                # in case of multiple pathology labels, use the worst label
                line_assessment = ASSESSMENT_MAP[line['pathology']]
                if subjects[scan_key]['assessment_label'] != line_assessment:
                    print_verbose(('upgrading %s != %s' % (subjects[scan_key]['assessment_label'], line_assessment)))
                    subjects[scan_key]['assessment_label'] = line_assessment
                    subjects[scan_key]['assessment_label'] = ASSESSMENTS[max(ASSESSMENTS.index(subjects[scan_key]['assessment_label']), ASSESSMENTS.index(line_assessment))]

                # in case of multiple BI-RADS assessments, use the highest score
                if subjects[scan_key]['assessment_score'] != int(line['assessment']):
                    print_verbose(('upgrading %d != %d' % (subjects[scan_key]['assessment_score'], int(line['assessment']))))
                    subjects[scan_key]['assessment_score'] = max(subjects[scan_key]['assessment_score'], int(line['assessment']))

                # check consistency
                assert subjects[scan_key]['patient_id'] == line['patient_id']
                assert subjects[scan_key]['side'] == SIDE_MAP[line['left or right breast']]
                assert subjects[scan_key]['view'] == VIEW_MAP[line['image view']]
                assert subjects[scan_key]['density_score'] == int(line['breast density'] if 'breast density' in line else line['breast_density'])
                assert subjects[scan_key]['image_file'] == image_file_path, \
                       ('%s != %s' % (subjects[scan_key]['image_file'], image_file_path))
                assert subjects[scan_key]['assessment_score'] >= int(line['assessment']), \
                       ('%d < %d' % (subjects[scan_key]['assessment_score'], int(line['assessment'])))

                subjects[scan_key]['mask_file'].append(mask_file_path)

#               if img_size != mask_size:
#                   print_verbose('  ' + image_file_path)
#                   print_verbose('    img:', img_size, img_size[0] / img_size[1])
#                   print_verbose('    crop:', crop_size, crop_size[0] / crop_size[1])
#                   print_verbose('    mask:', mask_size, mask_size[0] / mask_size[1])


if args.save_hdf5:
    h5out = h5py.File(args.save_hdf5, 'w')


for subject_idx, (subject_id, subject) in enumerate(tqdm.tqdm(subjects.items())):
    # load input image
    ds = pydicom.dcmread(subject['image_file'])
    image = ds.pixel_array
    image_fg_mask = image > 0

    print_verbose('Shape of input:', image.shape)
    print_verbose('Unique values: ', np.unique(image))
    print_verbose('Percentage > 0:', np.sum(image_fg_mask) / float(np.product(image.shape)))

    # scale roughly to [-1, +1]
    image = 2 * (image.astype(float) / 65534) - 1

    view = subject['view']
    side = subject['side']


    # load input segmentations
    print_verbose('Segmentations:')
    segmentations = []
    for i, s in enumerate(subject['mask_file']):
        ds = pydicom.dcmread(s)
        segmentation = ds.pixel_array

        # some segmentations have an incorrect size
        if segmentation.shape != image.shape:
            print_verbose('  %d: Original shape:' % i, segmentation.shape)
            segmentation = skimage.transform.resize(segmentation, image.shape, order=0)

        segmentations.append(segmentation > 0)
        print_verbose('  %d: Segmentation shape:' % i, segmentation.shape)
        assert segmentation.shape == image.shape
        assert len(np.unique(segmentation)) == 2
        print_verbose('      Foreground voxels: ', np.sum(segmentation > 0))
        print_verbose('      Unique values:     ', np.unique(segmentation))


    # compute combined segmentation overlay
    combined_segmentation = np.zeros(image.shape, dtype='bool')
    for segmentation in segmentations:
        combined_segmentation = np.logical_or(combined_segmentation, segmentation)

    # rescale
    if args.rescale is not None:
        print_verbose('  Rescale %0.2f' % args.rescale)
        image = skimage.transform.rescale(image, args.rescale, order=3).astype('float32')
        combined_segmentation = skimage.transform.rescale(combined_segmentation.astype('float32'),
                                                          args.rescale, order=0).astype('float32')
        print_verbose('    Foreground voxels:', np.sum(combined_segmentation > 0))

    def compute_patch_offset(centroid, patch_size, image_size):
        centroid = int(np.floor(centroid))
        offset = max(0, centroid - patch_size // 2)
        offset = min(offset, image_size - patch_size)
        return offset

    print_verbose('Finding crop')
    rprops = skimage.measure.regionprops((combined_segmentation > 0).astype(int))[0]
    # find bounding box
    bbox = rprops.bbox
    print_verbose('  Bbox size: ', bbox[2] - bbox[0], bbox[3] - bbox[1])
    # find centroid
    centroid = rprops.centroid
    print_verbose('  Centroid: ', centroid)


    if args.crop_size is None:
        print_verbose('  No cropping required')
        patch = image
        offsets = []
        crop_center = []
        crop_method = None
        crop_params = {}

    else:
        patch_size = (args.crop_size[0], args.crop_size[1])

        if args.crop_method == 'centroid':
            print_verbose('  Crop based on segmentation centroid')
            crop_center = centroid
            crop_method = 'centroid'
            crop_params = {}
        else:
            assert args.crop_method == 'nyu'
            # https://cs.nyu.edu/~kgeras/reports/datav1.0.pdf
            # find optimal center
            try:
                crop_center, crop_bbox, wininfo = \
                    nyu_cropping.find_crop_center(image, crop_size=patch_size,
                                                  side=side, view=view)
            except Exception as e:
                if args.verbose:
                    traceback.print_exc()
                print_verbose(subject['image_file'])
                print_verbose(e)
                print_verbose('  SKIP')
                continue
            print_verbose('  Crop based on nonzero mask (NYU method)')
            print_verbose('    Crop center: ', crop_center)
            crop_method = 'nyu'
            crop_params = {'crop_center': crop_center,
                           'crop_bbox': crop_bbox,
                           'crop_wininfo_fraction': wininfo['fraction']}
            print_verbose('    Crop params: ', crop_params)

        # crop image
        offsets = [compute_patch_offset(crop_center[0], patch_size[0], image.shape[0]),
                   compute_patch_offset(crop_center[1], patch_size[1], image.shape[1])]
        if patch_size[0] > image.shape[0] or patch_size[1] > image.shape[1]:
            # pad top and left
            pad_width = ((max(0, patch_size[0] - image.shape[0]), 0),
                         (max(0, patch_size[1] - image.shape[1]), 0))
            print_verbose('  Padding:  ', pad_width)
            offsets = [max(0, offsets[0]), max(0, offsets[1])]
            image = np.pad(image, pad_width, constant_values=np.min(image))
            combined_segmentation = np.pad(combined_segmentation, pad_width, constant_values=False)
        print_verbose('  Crop size:', patch_size)
        print_verbose('  Offsets:  ', offsets)

        patch = image[offsets[0]:offsets[0] + patch_size[0],
                      offsets[1]:offsets[1] + patch_size[1]]
        patch_segmentation = combined_segmentation[offsets[0]:offsets[0] + patch_size[0],
                                                   offsets[1]:offsets[1] + patch_size[1]]
        print_verbose('  Patch segmentation pixels:', patch_segmentation.sum())

    if args.save_hdf5:
        h5out.create_dataset('scans/%d/image' % subject_idx,
                             data=patch.astype('float16'), compression='gzip')
        h5out.create_dataset('scans/%d/segmentation' % subject_idx,
                             data=patch_segmentation.astype('bool'), compression='gzip')
        group = h5out['scans/%d' % subject_idx]
        for k in ('patient_id', 'side', 'view', 'density_score', 'assessment_score',
                  'assessment_label', 'lesion_type', 'image_file', 'mask_file'):
            group.attrs[k] = subject[k]
        group.attrs['subject_id'] = subject_id
        group.attrs['centroid'] = centroid
        if args.crop_size is not None:
            group.attrs['offsets'] = offsets
            group.attrs['crop_center'] = crop_center
            group.attrs['crop_method'] = crop_method
            for k, v in crop_params.items():
                group.attrs[k] = v

