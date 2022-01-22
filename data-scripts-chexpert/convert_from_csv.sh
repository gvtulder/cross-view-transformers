#!/bin/bash

# create new subset lists
# python -u random_split.py --csv /data/external/CheXpert/CheXpert-v1.0-small/train.csv --output-a lists/20210209-random-split-train.csv --output-b lists/20210209-random-split-val+test.csv --a-prop 0.75
# python -u random_split.py --csv lists/20210209-random-split-val+test.csv --output-a lists/20210209-random-split-val.csv --output-b lists/20210209-random-split-test.csv --a-prop 0.5

# convert images an create subsets
python -u convert_from_csv.py --csv lists/20210209-random-split-train.csv --basedir /data/external/CheXpert/ --save-hdf5 data/chexpert-small-20210209-custom-train.h5
python -u convert_from_csv.py --csv lists/20210209-random-split-val.csv   --basedir /data/external/CheXpert/ --save-hdf5 data/chexpert-small-20210209-custom-val.h5
python -u convert_from_csv.py --csv lists/20210209-random-split-test.csv  --basedir /data/external/CheXpert/ --save-hdf5 data/chexpert-small-20210209-custom-test.h5

