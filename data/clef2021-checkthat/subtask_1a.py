import pdb
import pandas as pd
import random
import numpy as np
import logging
import argparse
from os.path import join, dirname, basename

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import sys
sys.path.append('.')

from scorer.subtask_1a import evaluate
from format_checker.subtask_1a import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


# topic_id tweet_id score run_id
def run_random_baseline(data_fpath, results_fpath):
    gold_df = pd.read_csv(data_fpath, sep='\t')
    with open(results_fpath, "w") as results_file:
        for i, line in gold_df.iterrows():
            results_file.write('{}\t{}\t{}\t{}\n'.format(line['topic_id'], line['tweet_id'], 
                random.random(), "random"))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath):
    train_df = pd.read_csv(train_fpath, sep='\t')
    test_df = pd.read_csv(test_fpath, sep='\t')

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1))),
        ('clf', SVC(C=1, gamma=0.75, kernel='rbf', random_state=0))
    ])
    pipeline.fit(train_df['tweet_text'], train_df['check_worthiness'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.decision_function(test_df['tweet_text'])
        for i, line in test_df.iterrows():
            dist = predicted_distance[i]
            results_file.write("{}\t{}\t{}\t{}\n".format(line['topic_id'], line['tweet_id'], 
                dist, "ngram"))


def run_baselines(train_fpath, test_fpath, lang, subtask='1a'):
    random_baseline_fpath = join(ROOT_DIR, f'baselines/data/subtask_{subtask}_random_baseline_{lang}_{basename(test_fpath)}')
    run_random_baseline(test_fpath, random_baseline_fpath)
    if check_format(random_baseline_fpath):
        thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(test_fpath, random_baseline_fpath)
    logging.info(f"Random Baseline for Subtask-{subtask}--{lang} AVGP: {avg_precision}")

    ngram_baseline_fpath = join(ROOT_DIR, f'baselines/data/subtask_{subtask}_ngram_baseline_{lang}_{basename(test_fpath)}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath)
    if check_format(ngram_baseline_fpath):
        thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(test_fpath, ngram_baseline_fpath)
    logging.info(f"Ngram Baseline for Subtask-{subtask}--{lang} AVGP: {avg_precision}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['spanish', 'arabic', 'english', 'bulgarian'],
                        help="The language of the subtask")
    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path, args.lang, subtask='1a')
