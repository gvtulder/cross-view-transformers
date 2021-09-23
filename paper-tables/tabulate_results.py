import argparse
import os
import re
from collections import defaultdict

import tqdm
import numpy as np
import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# 20210121-model-TwoViewAttentionResNet18-attnheads-12-attndds-1-attncomb-add-data-ddsm-view-ccmlo-run1-aug-weight
KEY_PATTERN = re.compile('.*model-(?P<model>[^-]+)-(attnheads-(?P<attnheads>.+?)-)?(attndds-(?P<attnds>.+?)-'
                         'attncomb-(?P<attncomb>.+?)-)?data-(?P<data>.+)-view-(?P<view>.+)-'
                         'run(?P<run>[0-9]+)(-(?P<aug>aug-flip|aug))?.*')


def tabulate_events(run_dirs):
    lines = []

    for run_dir in tqdm.tqdm(run_dirs):
        experiment_id = run_dir.split('/')[-1]
        m = re.match(KEY_PATTERN, experiment_id)

        if m is None:
            print('Not matched:', run_dir)
            continue

        # load measurements
        it = EventAccumulator(run_dir).Reload()

        # which tags to optimize?
        optimize_tags = [tag for tag in it.Tags()['scalars'] if args.tag_to_optimize in tag]
        for optimize_tag in optimize_tags:
            # find the best epoch
            if args.direction == 'min':
                best_epoch = np.argmin([i.value for i in it.Scalars(optimize_tag)])
            else:
                best_epoch = np.argmax([i.value for i in it.Scalars(optimize_tag)])

            # start output
            line = {}
            line['experiment'] = experiment_id
            line.update(m.groupdict())
            line['optimize_tag'] = optimize_tag
            line['direction'] = args.direction
            line['best_epoch'] = best_epoch

            for tag in it.Tags()['scalars']:
                if args.tag_to_measure in tag:
                    # add to output
                    line[tag] = it.Scalars(tag)[best_epoch].value

            lines.append(line)

    return pd.DataFrame(lines)


parser = argparse.ArgumentParser()
parser.add_argument('--runs', metavar='DIR', required=True, nargs='+')
parser.add_argument('--direction', choices=('min', 'max'), default='max')
parser.add_argument('--tag-to-optimize', metavar='TAG', required=True)
parser.add_argument('--tag-to-measure', metavar='TAG', required=True)
parser.add_argument('--ipdb', action='store_true')
args = parser.parse_args()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = tabulate_events(args.runs)
# print(df)
# print()
# pivot = df.pivot(index=['experiment', 'model', 'attnheads', 'attnds', 'attncomb', 'data', 'view', 'run'],
print(df)
pivot = df.pivot(index=['model', 'attnheads', 'attnds', 'attncomb', 'data', 'view', 'aug', 'run'],
                 columns='optimize_tag',
                 values=['best_epoch', args.tag_to_measure])
rank = pivot[args.tag_to_measure].rank(ascending=False)
for c in rank.columns:
    pivot['rank', c] = rank[c]
print(pivot.to_csv())

if args.ipdb:
    import ipdb ; ipdb.set_trace()

