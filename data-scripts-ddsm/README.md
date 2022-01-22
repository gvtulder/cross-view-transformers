Cross-view transformers for multi-view analysis of unregistered medical images
==============================================================================
This code accompanies the paper

> Multi-view analysis of unregistered medical images using cross-view-transformers <br>
> by [Gijs van Tulder](https://vantulder.net/), Yao Tong, Elena Marchiori <br>
> from the [Data Science Group](https://www.ru.nl/das/), Faculty of Science, Radboud University, Nijmegen, the Netherlands <br>
> Presented at MICCAI 2021.

The paper is available at
* https://doi.org/10.1007/978-3-030-87199-4_10 (Springer)
* https://arxiv.org/abs/2103.11390

The most recent version of this code is available at https://vantulder.net/code/2021/miccai-transformers/

## Description

This directory contains the preprocessing scripts for the CBIS-DDSM data. See `create-cbis-ddsm-datasets.sh` for the steps and arguments used for the paper.

The scripts require the data from the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).

## Copyright and license

Copyright (C) 2021 [Gijs van Tulder](https://vantulder.net/) / Radboud University, the Netherlands

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Additional code

This directory also contains the `calc_optimal_centers.py` script from https://github.com/nyukat/breast_cancer_classifier/. See the copyright and license in `calc_optimal_centers.py`.
