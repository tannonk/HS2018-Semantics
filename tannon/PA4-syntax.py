#!/usr/bin/env python
# coding: utf-8

# Anastassia Shaitarova, Julia Nigmatunlina, Tannon Kew
# To Run: python3 PA4-no_syntax.py test_labels.txt


# In[4]:


import gzip
import numpy as np
import random
import os
import json
import sys
from collections import Counter, defaultdict, namedtuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import FunctionTransformer,LabelEncoder
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import networkx as nx
import spacy
nlp = spacy.load('en')

try:
    outfile = sys.argv[1]
except IndexError:
    print("Please specify name of outfile for labels.")
    sys.exit(0)


# In[5]:


##################################################################################################
# 1. LOAD DATA ALTERED
##################################################################################################

PairExample = namedtuple('PairExample',
    'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
    'left, mention_1, middle, mention_2, right, direction')
def load_data(file, verbose=True):
    f = open(file,'r', encoding='utf-8')
    data = []
    labels = []
    for i,line in enumerate(f):
        instance = json.loads(line)

        instance_tuple = PairExample(instance['entity_1'],instance['entity_2'],[])

        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'],snippet['mention_1'],snippet['middle'],
                                   snippet['mention_2'],snippet['right'],
                                    snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)

                data.append(instance_tuple)
                labels.append(instance['relation'])
                instance_tuple = PairExample(instance['entity_1'],instance['entity_2'],[])

            except:
                print(instance)

    f.close()
    return data,labels

train_data, train_labels = load_data('../data/train.json.txt')

print('*'*40)
print("Training data successfully loaded.")
print("{} sample in train_data".format(len(train_data)))
print('*'*40)

# In[6]:


# Statistics over relations
def print_stats(labels):
    labels_counts = Counter(labels)
    print('{:20s} {:>10s} {:>10s}'.format('', '', 'rel_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('relation', 'examples', '/all_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    for k,v in labels_counts.items():
        print('{:20s} {:10d} {:10.2f}'.format(k, v, v /len(labels)))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    print('{:20s} {:10d} {:10.2f}'.format('Total', len(labels), len(labels) /len(labels)))

print('Train set statistics:')
print_stats(train_labels)
print('*'*40)

# In[7]:


## check that each entity pair is assigned only one relation
# pair_dict={}
# rel_dict={}
# for example, label in zip(train_data,train_labels):
#     if (example.entity_1,example.entity_2) not in pair_dict.keys():
#         pair_dict[(example.entity_1,example.entity_2)] = [label]

#     else:
#         pair_dict[(example.entity_1,example.entity_2)].append(label)
# #         print(example.entity_1,example.entity_2,label)
#     if label not in rel_dict.keys():
#         rel_dict[label] = [example]
#     else:
#         rel_dict[label].append(example)
# print("Done building dictionary")

# # example for each relation
# for rel in rel_dict.keys():
#     ex = rel_dict[rel][0]
#     print(rel,ex.entity_1,ex.entity_2)


# In[8]:


def SelectContext(data, verbose=True):
    """BOW feature extraction"""
    only_context_data = []
    for instance in data:

        instance_context = []
        for s in instance.snippet:
            context = s.left + " m_1 " + s.middle + " m_2 " + s.right
            instance_context.append(context)
        only_context_data.append(' '.join(instance_context))
    if verbose:
        print(len(only_context_data))
        print(only_context_data[0])
        print(only_context_data[0])
    return only_context_data


# In[186]:


# test_feat = SelectContext(train_data[:200])


# In[9]:


def ExractSimpleFeatures(data, verbose=True):
    """Considers length and words of middle segment"""
    featurized_data = []
    for instance in data:
        featurized_instance = {'mid_words': '', 'distance': np.inf}
        for s in instance.snippet:
            if len(s.middle.split()) < featurized_instance['distance']:
                featurized_instance['mid_words'] = s.middle
                featurized_instance['distance'] = len(s.middle.split())
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(featurized_data))
        print(featurized_data[0])
        print(featurized_data[1])
    return featurized_data


# In[12]:


# test_feat = ExractSimpleFeatures(train_data[:200])


# In[47]:


def LengthOfEntities(data, verbose=True):
    featurized_data = []
    for instance in data:
        featurized_instance = {
            'entity1_len': len(instance.entity_1.split("_")),
            'entity2_len': len(instance.entity_2.split("_")),
            'combined_len': len(instance.entity_1.split("_")) + len(instance.entity_2.split("_"))
        }
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(featurized_data))
        print(featurized_data[0])
        print(featurized_data[1])
    return featurized_data


# In[48]:


# test_feat = LengthOfEntities(train_data[:200])


# In[49]:


def UseNLP(data, verbose=True):
    """
    Processes each data instance with Spacy's nlp pipeline
    and collects POS tags for context window around mentions
    as well as length of dependency path between two mentions
    """
    featurized_data = []

    for instance in data:
        featurized_instance = {'tagged_context_1': '', 'tagged_context_2': '', 'path_length': 0}

        for s in instance.snippet:

            context = s.left + " m_1 " + s.middle + " m_2 " + s.right

            document = nlp(context) # spacy pipeline

            tagged_context_1 = []
            tagged_context_2 = []

            for i, w in enumerate(document):
                if w.orth_ == "m_1":
                    window_1 = document[i-3:i+4]
                    for e in window_1:
                        if e.orth_ == "m_1" or e.orth_ == "m_2":
                            tagged_context_1.append("MENTION")
                        else:
                            tagged_context_1.append(e.pos_)

                if w.orth_ == "m_2":
                    window_2 = document[i-3:i+4]
                    if window_2:
                        for e in window_2:
                            if e.orth_ == "m_1" or e.orth_ == "m_2":
                                tagged_context_2.append("MENTION")
                            else:
                                tagged_context_2.append(e.pos_)

            featurized_instance['tagged_context_1'] = ' '.join(tagged_context_1)
            featurized_instance['tagged_context_2'] = ' '.join(tagged_context_2)

            edges = []
            for w in document: # FYI https://spacy.io/docs/api/token
                for child in w.children:
                    edges.append(('{0}-{1}'.format(w.lower_, w.i),
                                  '{0}-{1}'.format(child.lower_, child.i)))

            graph = nx.Graph(edges)
            for w in graph:
                if "m_1" in w:
                    s = w
                if "m_2" in w:
                    t = w

            try:
                featurized_instance['path_length'] = nx.shortest_path_length(graph, source=s, target=t)
            except nx.NetworkXNoPath: # unrelated?
                featurized_instance['path_length'] = 0
            except nx.NodeNotFound: # problem with mention
                featurized_instance['path_length'] = 0.5

        featurized_data.append(featurized_instance)

        if len(featurized_data)%5000 == 0:
            print("{} instances processed for nlp.".format(len(featurized_data)))

    if verbose:
        print(len(featurized_data))
        print(featurized_data[0])
        print(featurized_data[1])

    return featurized_data


# In[50]:


# test_feat = UseNLP(train_data[:500])


# In[51]:


class SimpleFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, featurizer):
        self.featurizers = featurizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return ExractSimpleFeatures(X, verbose=False)


# In[52]:


class EntityLengthFeaturizer(BaseEstimator, TransformerMixin):
    """Extract features from each isntance for DictVectorizer"""
    def __init__(self, featurizer):
        self.featurizers = featurizer

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return LengthOfEntities(X, verbose=False)


# In[53]:


class BowFeaturizer(BaseEstimator, TransformerMixin):
    """BOW featurizer"""
    def __init__(self, featurizer):
        self.featurizers = featurizer

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return SelectContext(X, verbose=False)


# In[54]:


class DependencyPath(BaseEstimator, TransformerMixin):
    """Considers pos tags of window around mentions and length of dependency path between mentions"""
    def __init__(self, featurizer):
        self.featurizers = featurizer

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # Applies spacy's nlp pipeline to data. Status indicates how many instances have been processed.
        return UseNLP(X, verbose=False)


# In[55]:


# Transform labels to numeric values
le = LabelEncoder()
train_labels_featurized = le.fit_transform(train_labels)

length_pipe = make_pipeline(EntityLengthFeaturizer(LengthOfEntities), DictVectorizer())

bow_pipe = make_pipeline(BowFeaturizer(SelectContext), CountVectorizer(ngram_range=(1,3)))

simple_pipe = make_pipeline(SimpleFeaturizer(ExractSimpleFeatures), DictVectorizer())

syntax_pipe = make_pipeline(DependencyPath(UseNLP), DictVectorizer())


clf = make_pipeline(FeatureUnion(transformer_list=[
    ('length_pipeline', length_pipe),
    ('bow_pipeline', bow_pipe),
    ('simple_pipeline', simple_pipe),
    ('syntax_pipeline', syntax_pipe)]),
    LogisticRegression())


# In[56]:


##################################################################################################
# 3. TRAIN CLASSIFIER AND EVALUATE (CV)
##################################################################################################

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
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format('macro-average', *avg_result))

def macro_average_results(results):
    avg_result = [np.average([r[i] for r in results.values()]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results.values()]))
    return avg_result

def average_results(results):
    avg_result = [np.average([r[i] for r in results]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results]))
    return avg_result

def evaluateCV(classifier, label_encoder, X, y, verbose=True):
    """
    classifier: clf - pipeline with CountVevtorizer and Logistic regression
    label_encoder: le - label encoder
    X: train data featurized
    y: train labels featurized
    """
    results = {}
    for rel in le.classes_:
#         print(rel)
        results[rel] = []
    if verbose:
        print_statistics_header()
        kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
        for train_index, test_index in kfold.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            clf.fit(X_train, y_train)
            pred_labels = classifier.predict(X_test)
            stats = precision_recall_fscore_support(y_test, pred_labels, beta=0.5)
            #print(stats)
            for rel in label_encoder.classes_:
                rel_id = label_encoder.transform([rel])[0]
#             print(rel_id,rel)
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


# In[57]:


evaluateCV(clf, le, train_data, train_labels_featurized)

print('*'*40)


# In[ ]:


# A check for the average F1 score

f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')

def evaluateCV_check(classifier, X, y, verbose=True):
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring = f_scorer)
    print("\nCross-validation scores (StratifiedKFold): ", scores)
    print("Mean cv score (StratifiedKFold): ", scores.mean())


# In[ ]:


evaluateCV_check(clf, train_data, train_labels_featurized)
print('*'*40)


# In[ ]:


##################################################################################################
# 4. TEST PREDICTIONS and ANALYSIS
##################################################################################################

# Fit final model on the full train data
clf.fit(train_data, train_labels_featurized)

# Predict on test set
test_data, test_labels = load_data('../data/test-covered.json.txt', verbose=False)
print(len(test_data))
print(len(test_labels))
# test_data_featurized = SelectContext(test_data, verbose=False)
test_label_predicted = clf.predict(test_data)
print(len(test_label_predicted))
# Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
print(len(test_label_predicted_decoded))
print(test_label_predicted_decoded[:2])
f = open("outputs/test_labels2.txt", 'w', encoding="utf-8")
for label in test_label_predicted_decoded:
    f.write(label+'\n')


# In[ ]:


# # Feature analysis - print N most informative
# # !! Make changes in this function when you change the pipleine!!
# def printNMostInformative(classifier,label_encoder,N):
#     """Prints features with the highest coefficient values, per class"""
#     feature_names = classifier.named_steps['countvectorizer'].get_feature_names()

#     coef = classifier.named_steps['logisticregression'].coef_
#     print(coef.shape)
#     for rel in label_encoder.classes_:
#         rel_id = label_encoder.transform([rel])[0]
#         coef_rel = coef[rel_id]
#         coefs_with_fns = sorted(zip(coef_rel, feature_names))
#         top_features = coefs_with_fns[-N:]
#         print("\nClass {} best: ".format(rel))
#         for feat in top_features:
#             print(feat)

# print("Top features used to predict: ")
# # show the top features
# printNMostInformative(clf,le,2)


# In[ ]:

# Not sure how to get this function to work with out pipeline at the moment.

# Feature analysis - print N most informative
# !! Make changes in this function when you change the pipleine!!
# def printNMostInformative(classifier,label_encoder,N):
#     """Prints features with the highest coefficient values, per class"""
#     feature_names = classifier.named_steps['length_pipeline'].get_feature_names()
#
#     coef = classifier.named_steps['logisticregression'].coef_
#     print(coef.shape)
#     for rel in label_encoder.classes_:
#         rel_id = label_encoder.transform([rel])[0]
#         coef_rel = coef[rel_id]
#         coefs_with_fns = sorted(zip(coef_rel, feature_names))
#         top_features = coefs_with_fns[-N:]
#         print("\nClass {} best: ".format(rel))
#         for feat in top_features:
#             print(feat)
#
# print("Top features used to predict: ")
# # show the top features
# printNMostInformative(clf,le,2)

###############################################################################
## End Code
