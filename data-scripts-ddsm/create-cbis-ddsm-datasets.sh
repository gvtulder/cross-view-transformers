#!/bin/bash

# convert original CBIS-DDSM data
# (downloaded in the data/CBIS-DDSM directory)
for view in CC MLO ; do
  for subset in train test ; do
    # crop to 90th percentile bounding box
    python -u convert_from_csv.py \
      --csv data/CBIS-DDSM/lists/mass_case_description_${subset}_set.csv \
      --basedir data/CBIS-DDSM \
      --rescale 0.5 \
      --crop-size 2438 1504 \
      --crop-method nyu \
      --save-hdf5 data/ddsm-csv-rescale0.5-nyucrop-mass-$view-${subset}.h5 \
      --only-view $view
  done
done

# create CC and MLO pairs
for subset in train test ; do
  python -u merge_views.py \
    --source-cc  data/ddsm-csv-rescale0.5-nyucrop-mass-CC-$subset.h5 \
    --source-mlo data/ddsm-csv-rescale0.5-nyucrop-mass-MLO-$subset.h5 \
    --save-hdf5  data/ddsm-csv-rescale0.5-nyucrop-mass-CC+MLO-$subset.h5
done

# split CBIS-DDSM in five subsets
python -u create_five_subsets.py \
  --source data/ddsm-csv-rescale0.5-nyucrop-mass-CC+MLO-train.h5 \
           data/ddsm-csv-rescale0.5-nyucrop-mass-CC+MLO-test.h5 \
  --save-hdf5-base data/20210219-ddsm-csv-rescale0.5-nyucrop-mass-CC+MLO

# downsample
for t in 0 1 2 3 4 ; do
  python -u downsample_hdf5.py \
    --input data/20210219-ddsm-csv-rescale0.5-nyucrop-mass-CC+MLO-subset$t.h5 \
    --output data/20210219-ddsm-csv-rescale0.5-nyucrop-ds0.125-mass-CC+MLO-subset$t.h5 \
    --scale 0.125
done

