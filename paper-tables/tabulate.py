import argparse
import os
import json
import queue
from collections import defaultdict
import multiprocessing.pool

import tqdm
import numpy as np
import pandas as pd
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run_dir(run_dir, tag_to_optimize, tag_to_measure, direction):
    lines = []
    try:
        # load measurements
        it = EventAccumulator(run_dir).Reload()

        # the arguments should be stored as a json strong
        summary_event = it.Tensors('args/text_summary')[0]
        experiment_args = json.loads(summary_event.tensor_proto.string_val[0].decode('utf-8'))

        # which tags to optimize?
        optimize_tags = [tag for tag in it.Tags()['scalars'] if tag_to_optimize in tag]
        for optimize_tag in optimize_tags:
            # find the best epoch
            if direction == 'min':
                best_epoch = np.argmin([i.value for i in it.Scalars(optimize_tag)])
            elif direction == 'max':
                best_epoch = np.argmax([i.value for i in it.Scalars(optimize_tag)])
            elif direction == 'last':
                best_epoch = len([i.value for i in it.Scalars(optimize_tag)]) - 1

            # start output
            line = {}
            line['run_dir'] = run_dir
            line.update(experiment_args)
            line['optimize_tag'] = optimize_tag
            line['direction'] = direction
            line['best_epoch'] = best_epoch

            for tag in it.Tags()['scalars']:
                if any(t in tag for t in tag_to_measure):
                    # add to output
                    line[tag] = it.Scalars(tag)[best_epoch].value

            lines.append(line)
    except KeyError as e:
        print(run_dir)
        print(e)
    return lines


def process(queue_in, queue_out, tag_to_optimize, tag_to_measure, direction):
    while True:
        try:
            run_dir = queue_in.get(False, 5000)
        except queue.Empty:
            print('Empty!')
            return
        print(run_dir)
        lines = load_run_dir(run_dir, tag_to_optimize, tag_to_measure, direction)
        queue_out.put(lines)

def load_fn(args):
    return load_run_dir(*args)

def tabulate_events(run_dirs, tag_to_optimize, tag_to_measure, direction):
    pool = multiprocessing.pool.Pool()
    arglist = [(run_dir, tag_to_optimize, tag_to_measure, direction) for run_dir in run_dirs]

    lines = []
    for l in tqdm.tqdm(pool.imap(load_fn, arglist), total=len(arglist)):
        lines += l

    return pd.DataFrame(lines)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', metavar='DIR', required=True, nargs='+')
    parser.add_argument('--direction', choices=('min', 'max', 'last'), default='max')
    parser.add_argument('--tag-to-optimize', metavar='TAG', required=True)
    parser.add_argument('--tag-to-measure', metavar='TAG', required=True, nargs='+')
    parser.add_argument('--save-csv', metavar='CSV')
    parser.add_argument('--ipdb', action='store_true')
    args = parser.parse_args()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = tabulate_events(args.runs, args.tag_to_optimize, args.tag_to_measure, args.direction)
    # print(df)
    # print()
    # pivot = df.pivot(index=['experiment', 'model', 'attnheads', 'attnds', 'attncomb', 'data', 'view', 'run'],
    print(df)
    print(df.to_csv())

    if args.save_csv:
        df.to_csv(args.save_csv)
