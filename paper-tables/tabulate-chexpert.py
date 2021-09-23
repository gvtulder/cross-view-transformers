import argparse
import os
import json
from collections import defaultdict

import tqdm
import numpy as np
import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASKS = ('Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
         'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')

def tabulate_events(run_dirs, direction):
    lines = []

    for run_dir in tqdm.tqdm(run_dirs):
        try:
            print(run_dir)
            # load measurements
            it = EventAccumulator(run_dir).Reload()

            # the arguments should be stored as a json strong
            summary_event = it.Tensors('args/text_summary')[0]
            experiment_args = json.loads(summary_event.tensor_proto.string_val[0].decode('utf-8'))

            # optimize for each task
            for task in TASKS:
                # compute the tag
                tag_to_optimize = 'roc_auc_task/val/%s' % task
                tag_to_measure = 'roc_auc_task/test/%s' % task
                
                # find the best epoch
                if direction == 'min':
                    best_epoch = np.argmin([i.value for i in it.Scalars(tag_to_optimize)])
                elif direction == 'max':
                    best_epoch = np.argmax([i.value for i in it.Scalars(tag_to_optimize)])
                elif direction == 'last':
                    best_epoch = len([i.value for i in it.Scalars(tag_to_optimize)]) - 1

                # start output
                line = {}
                line['run_dir'] = run_dir
                line.update(experiment_args)
                line['direction'] = direction
                line['task'] = task
                line['best_epoch'] = best_epoch

                # add measurements
                line['roc_auc_task/val'] = it.Scalars(tag_to_optimize)[best_epoch].value
                line['roc_auc_task/test'] = it.Scalars(tag_to_measure)[best_epoch].value

                lines.append(line)
        except KeyError as e:
            print(run_dir)
            print(e)

    return pd.DataFrame(lines)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', metavar='DIR', required=True, nargs='+')
    parser.add_argument('--direction', choices=('min', 'max', 'last'), default='max')
    parser.add_argument('--save-csv', metavar='CSV')
    parser.add_argument('--ipdb', action='store_true')
    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = tabulate_events(args.runs, args.direction)
    # print(df)
    # print()
    # pivot = df.pivot(index=['experiment', 'model', 'attnheads', 'attnds', 'attncomb', 'data', 'view', 'run'],
    print(df)
    print(df.to_csv())

    if args.save_csv:
        df.to_csv(args.save_csv)

