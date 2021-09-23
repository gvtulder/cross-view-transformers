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

This code implements cross-view transformers and the baseline networks evaluated in the paper.

## Structure

* The main directory contains the Python code for the models, training, and evaluation.
* The `experiments/` directory contains bash scripts with the experimental settings used in the paper.
* The `paper-tables/` directory contains results and scripts to generate the tables for the paper.

## Requirements

* The experiments were run on Python 3.6.9 with PyTorch 1.7.0.
* The data from the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) datasets is not included here, but can be downloaded elsewhere.

## Copyright and license

Copyright (C) 2021 [Gijs van Tulder](https://vantulder.net/) / Radboud University, the Netherlands

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

