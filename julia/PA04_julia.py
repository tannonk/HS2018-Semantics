#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SA, HS18
# PA04


import gzip
import numpy as np
import random
import os
import json
from pathlib import Path
from scipy.sparse import csr_matrix

from collections import Counter, defaultdict, namedtuple
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.models import Word2Vec
import spacy

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from sklearn.feature_selection import SelectFwe, SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')

# LOAD THE DATA
PairExample = namedtuple('PairExample',
                         'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
                     'left, mention_1, middle, mention_2, right, direction')


def load_data(file, verbose=False):
    ''' Loads the data from .json file.
        Returns: labels and data as separate variables.
    '''
    f = open(file, 'r', encoding='utf-8')
    data = []
    labels = []
    for i, line in enumerate(f):
        instance = json.loads(line)
        if i == 0:
            if verbose:
                print('json example:')
                print(instance)
        # 'relation, entity_1, entity_2, snippet' fileds for each example
        # 'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'], instance['entity_2'], [])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'], snippet['mention_1'],
                                        snippet['middle'],
                                        snippet['mention_2'], snippet['right'],
                                        snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i == 0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])

    return data, labels


def print_stats(labels):
    ''' Returns: statistics over relations
    '''
    labels_counts = Counter(labels)
    print('{:20s} {:>10s} {:>10s}'.format('', '', 'rel_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('relation', 'examples',
                                          '/all_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))

    for k,v in labels_counts.items():
        print('{:20s} {:10d} {:10.2f}'.format(k, v, v / len(labels)))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    print('{:20s} {:10d} {:10.2f}'.format('Total', len(labels),
                                          len(labels) / len(labels)))


# get full context
def get_context(data, embed_mode=False):
    all_data = []
    for instance in data:
        s_context = []
        for s in instance.snippet:
            if embed_mode:
                s_context.append(' '.join((s.left, s.mention_1.replace(" ", "_"),
                                           s.middle, s.mention_2.replace(" ", "_"),
                                           s.right)))
            else:
                # s_context.append(' '.join((s.left, s.mention_1, s.middle, s.mention_2, s.right)))
                s_context.append(' '.join((s.left, s.middle, s.right)))
        all_data.append(' '.join(s_context))

    # print(len(all_data))
    return all_data


def ExractSimpleFeatures(data, verbose=False):
    ''' Extract simple features.
    '''
    featurized_data = []
    for instance in data:
        featurized_instance = {
            'mid_words': '',
            'distance': np.inf,
            'left': [],
            'right': [],
            'mid': []
        }
        for s in instance.snippet:
            if len(s.middle.split()) < featurized_instance['distance']:
                featurized_instance['mid_words'] = s.middle
                featurized_instance['distance'] = len(s.middle.split())
            featurized_instance['left'] = s.left
            featurized_instance['right'] = s.right
            featurized_instance['mid'] = s.middle
            # context = [s.left + s.right + s.middle]
            # vec_context = vectorizer.transform(context)
            # featurized_instance['left'] = vectorizer.transform([s.left])
            # featurized_instance['right'] = vectorizer.transform([s.right])
            # featurized_instance['mid'] = vectorizer.transform([s.middle])
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(data))
        print(len(featurized_data))
        print(data[0])
        print(featurized_data[0])

    return featurized_data


def train_word2vec(data=None, pretrained=True):
    ''' Input: list of contexts (each context is a string).
        Prepares data for training embeddings: tokenize with simple_preprocessing.
        Returns: embedding model
    '''
    # path_model = Path("models") / "Word2Vec.model"

    # if path_model.exists():
    #     model = Word2Vec.load(str(path_model))
    # else:
    #     if not path_model.parent.exists():
    #         path_model.parent.mkdir(parents=True)
    if pretrained:
        model = api.load("glove-wiki-gigaword-100")
    else:
        data_tokenised = [doc.lower().split(" ") for doc in data]
        model = Word2Vec(data_tokenised, size=100, min_count=1, sg=1)
        # model.save(str(path_model))

    return model


def extract_embeddings_feature(data, embed_model, verbose=True):
    ''' compute the average values of word embedding for words of each instance
    '''
    DIMEN_SIZE = 100
    vectorized_data = np.zeros(shape=(len(data), DIMEN_SIZE))

    for i, instance in enumerate(data):
        instance_vector = []
        for word in instance.split(" "):
            if embed_model.wv.vocab.get(word, None) is not None:
                word_vec = embed_model.wv.get_vector(word)
            else:
                word_vec = np.random.randn(embed_model.vector_size)
            instance_vector.append(word_vec)

        if len(instance_vector) > 0:
            instance_array = np.array(instance_vector)
            avg_vector = np.mean(instance_array, axis=0)
        else:
            avg_vector = np.zeros(embed_model.vector_size)

        vectorized_data[i] = avg_vector
        assert not np.isnan(avg_vector).any()

    if verbose:
        print(len(data))
        print(len(vectorized_data))

    return vectorized_data


def print_statistics_header():
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        'relation', 'precision', 'recall', 'f-score', 'support'))
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))


def print_statistics_row(rel, result):
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format(rel, *result))


def print_statistics_footer(avg_result):
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format('macro-average',
                                                            *avg_result))


def macro_average_results(results):
    avg_result = [np.average([r[i] for r in results.values()]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results.values()]))
    return avg_result


def average_results(results):
    avg_result = [np.average([r[i] for r in results]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results]))
    return avg_result


def evaluateCV(classifier, label_encoder, X, y, verbose=True):
    results = {}
    for rel in label_encoder.classes_:
        results[rel] = []
    if verbose:
        print_statistics_header()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_index, test_index in kfold.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            classifier.fit(X_train, y_train)
            pred_labels = classifier.predict(X_test)
            stats = precision_recall_fscore_support(y_test, pred_labels, beta=0.5)
            # print(stats)
            for rel in label_encoder.classes_:
                rel_id = label_encoder.transform([rel])[0]
            # print(rel_id,rel)
                stats_rel = [stat[rel_id] for stat in stats]
                results[rel].append(stats_rel)
        for rel in label_encoder.classes_:
            results[rel] = average_results(results[rel])
            if verbose:
                print_statistics_row(rel, results[rel])
    avg_result = macro_average_results(results)
    if verbose:
        print_statistics_footer(avg_result)
    return avg_result[2]  # return f_0.5 score as summary statistic


def evaluateCV_check(classifier, X, y, verbose=True):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring=f_scorer)
    print("\nCross-validation scores (StratifiedKFold): ", scores)
    print("Mean cv score (StratifiedKFold): ", scores.mean())


def main():
    train_data, train_labels = load_data('data/train.json.txt')
    print('Train set statistics:')
    print_stats(train_labels)
    test_data, test_labels = load_data('data/test.json.txt', verbose=False)

    all_train = get_context(train_data, embed_mode=True)
    # all_test = get_context(test_data, embed_mode=True)

    # DATA ExractSimpleFeatures
    # train_simple_featurized = ExractSimpleFeatures(train_data, verbose=False)
    # test_simple_featurized = ExractSimpleFeatures(test_data, verbose=False)

    # Transform labels to nimeric values
    le = LabelEncoder()
    train_labels_featurized = le.fit_transform(train_labels)

    # Fit model one vs rest logistic regression
    # clf = make_pipeline(DictVectorizer(), LogisticRegression())

    # if with CountVectorizer
    # bow_vectorizer = CountVectorizer(ngram_range=(1, 3),
    #                                  # max_df=0.9
    #                                  )
    # TFiDF_vectorizer = TfidfVectorizer()

    # With Word embeddings:
    embed_model = train_word2vec()

    LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                            fit_intercept=True, intercept_scaling=1,
                            class_weight=None, random_state=None,
                            solver='liblinear', max_iter=100, multi_class='ovr',
                            verbose=0, warm_start=False, n_jobs=None)

    X_train = extract_embeddings_feature(all_train, embed_model, verbose=True)

    # clf = make_pipeline(bow_vectorizer, LR)
    clf = LR

    # CV
    # evaluateCV(clf, le, X_train, train_labels_featurized)
    evaluateCV_check(clf, X_train, train_labels_featurized)

    # TEST data
    # Fit final model on the full train data
    # clf.fit(all_train, train_labels_featurized)

    # Predict on test set
    # test_label_predicted = clf.predict(all_test)

    # Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
    # test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
    # print(test_label_predicted_decoded[:5])
    # f = open("test_labels.txt", 'w', encoding="utf-8")
    # for label in test_label_predicted_decoded:
    #     f.write(label + '\n')


if __name__ == "__main__":
    main()
