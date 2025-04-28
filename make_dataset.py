#!/usr/bin/env python3

# Create HF dataset from Nemotron-CC data

import sys
import os

from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets, ClassLabel


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('data_dir', help='directory with high-actual.jsonl etc.')
    ap.add_argument('repo_id', help='format: username/datasetname')
    return ap


CLASS_LABEL = ClassLabel(names=['low', 'medium-low', 'medium', 'medium-high', 'high'])


FILE_LABEL_MAP = {
    'low-actual.jsonl': 0,
    'medium-low-actual.jsonl': 1,
    'medium-actual.jsonl': 2,
    'medium-high-actual.jsonl': 3,
    'high-actual.jsonl': 4,
}


def main(argv):
    args = argparser().parse_args(argv[1:])

    labeled_datasets = []
    for fn, label in FILE_LABEL_MAP.items():
        path = os.path.join(args.data_dir, fn)
        d = load_dataset('json', data_files=path, split='all')
        print(f'label {label}: loaded {len(d)} examples from {fn}')
        d = d.add_column('label', [label]*len(d))
        labeled_datasets.append(d)

    combined = concatenate_datasets(labeled_datasets)
    combined = combined.cast_column('label', CLASS_LABEL)
    combined = combined.shuffle(seed=args.seed)

    combined.push_to_hub(args.repo_id)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
