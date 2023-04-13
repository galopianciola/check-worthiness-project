import pdb
import pandas as pd
import random
import numpy as np
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import sys
sys.path.append('.')

from scorer.subtask_1 import evaluate
from format_checker.subtask_1 import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def run_majority_baseline(data_fpath, test_fpath, results_fpath):
    train_df = pd.read_csv(data_fpath, sep='\t')
    test_df = pd.read_csv(test_fpath, sep='\t')

    pipeline = DummyClassifier(strategy="most_frequent")
    pipeline.fit(train_df['tweet_text'], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df['tweet_text'])
        results_file.write("topic\ttweet_id\tclass_label\trun_id\n")
        for i, line in test_df.iterrows():
            label = predicted_distance[i]
            results_file.write("{}\t{}\t{}\t{}\n".format(line['topic'], line['tweet_id'],label, "ngram"))

# topic_id tweet_id score run_id
def run_random_baseline(data_fpath, results_fpath):
    gold_df = pd.read_csv(data_fpath, sep='\t')
    label_list=gold_df['class_label'].to_list()

    with open(results_fpath, "w") as results_file:
        results_file.write("topic\ttweet_id\tclass_label\trun_id\n")
        for i, line in gold_df.iterrows():
            results_file.write('{}\t{}\t{}\t{}\n'.format(line['topic'], line['tweet_id'],random.choice(label_list), "random"))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath):
    train_df = pd.read_csv(train_fpath, sep='\t')
    test_df = pd.read_csv(test_fpath, sep='\t')

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1),lowercase=True,use_idf=True,max_df=0.95, min_df=3,max_features=5000)),
        ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
    ])
    pipeline.fit(train_df['tweet_text'], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df['tweet_text'])
        results_file.write("topic\ttweet_id\tclass_label\trun_id\n")
        for i, line in test_df.iterrows():
            label = predicted_distance[i]
            results_file.write("{}\t{}\t{}\t{}\n".format(line['topic'], line['tweet_id'],label, "ngram"))


def run_baselines(train_fpath, test_fpath, lang, subtask='checkworthy'):
    random_majority_fpath = join(ROOT_DIR,
                                 f'data/subtask_{subtask}_majority_baseline_{lang}_{basename(test_fpath)}')
    run_majority_baseline(train_fpath, test_fpath, random_majority_fpath)

    if check_format(random_majority_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, random_majority_fpath)
    # logging.info(f"Random Baseline for Subtask-{subtask}--{lang} AVGP: {avg_precision}")
    if (subtask == "checkworthy" or subtask == "harmful"):
        logging.info(f"Majority Baseline for Subtask-{subtask}--{lang} F1 (positive class): {f1}")
    elif (subtask == "attentionworthy"):
        logging.info(f"Majority Baseline for Subtask-{subtask}--{lang} F1 (Weighted): {f1}")
    elif (subtask == "claim"):
        logging.info(f"Majority Baseline for Subtask-{subtask}--{lang} Acc: {acc}")


    random_baseline_fpath = join(ROOT_DIR, f'data/subtask_{subtask}_random_baseline_{lang}_{basename(test_fpath)}')
    run_random_baseline(test_fpath, random_baseline_fpath)

    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, random_baseline_fpath)
    # logging.info(f"Random Baseline for Subtask-{subtask}--{lang} AVGP: {avg_precision}")
    if (subtask == "checkworthy" or subtask == "harmful"):
        logging.info(f"Random Baseline for Subtask-{subtask}--{lang} F1 (positive class): {f1}")
    elif (subtask == "attentionworthy"):
        logging.info(f"Random Baseline for Subtask-{subtask}--{lang} F1 (Weighted): {f1}")
    elif (subtask == "claim"):
        logging.info(f"Random Baseline for Subtask-{subtask}--{lang} Acc: {acc}")

    ngram_baseline_fpath = join(ROOT_DIR, f'data/subtask_{subtask}_ngram_baseline_{lang}_{basename(test_fpath)}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath)
    if check_format(ngram_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, ngram_baseline_fpath)
    if (subtask == "checkworthy" or subtask == "harmful"):
        logging.info(f"Ngram Baseline for Subtask-{subtask}--{lang} F1 (positive class): {f1}")
    elif (subtask == "attentionworthy"):
        logging.info(f"Ngram Baseline for Subtask-{subtask}--{lang} F1 (Weighted): {f1}")
    elif (subtask == "claim"):
        logging.info(f"Ngram Baseline for Subtask-{subtask}--{lang} Acc: {acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--test-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'bulgarian', 'dutch','english', 'spanish', 'turkish'],
                        help="The language of the subtask")
    parser.add_argument("--subtask", "-a", required=True, type=str,
                        choices=['checkworthy', 'claim', 'harmful','attentionworthy'],
                        help="The name of the subtask")

    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path, args.lang, subtask=args.subtask)


    #python -t /home/bjit/Nandi-AI/NLP/clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/CT22_english_1B_claim_train.tsv -d /home/bjit/Nandi-AI/NLP/clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/CT22_english_1B_claim_dev.tsv -l english -a claim
