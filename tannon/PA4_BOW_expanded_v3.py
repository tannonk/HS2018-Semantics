#!/usr/bin/env python
# coding: utf-8

# ### this script sets a baseline for relation extraction using frequency-based BOW model
# 
# #### add additional features

# In[2]:


import gzip
import numpy as np
import random
import os
import json

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

##### additional imports
import spacy
nlp = spacy.load('en')


# In[3]:


##################################################################################################
# 1. LOAD DATA
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
        if i==0:
            if verbose:
                print('json example:')
                print(instance)
        #'relation, entity_1, entity_2, snippet' fileds for each example
        #'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'],instance['entity_2'],[])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'],snippet['mention_1'],snippet['middle'],
                                   snippet['mention_2'],snippet['right'],
                                    snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i==0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])
    return data,labels
    
train_data, train_labels = load_data('../data/train.json.txt')


# In[4]:


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


# In[5]:


# check that each entity pair is assigned only one relation
pair_dict={}
rel_dict={}
for example, label in zip(train_data,train_labels):
    if (example.entity_1,example.entity_2) not in pair_dict.keys():
        pair_dict[(example.entity_1,example.entity_2)] = [label]
        
    else:
        pair_dict[(example.entity_1,example.entity_2)].append(label)
        print(example.entity_1,example.entity_2,label)
    if label not in rel_dict.keys():
        rel_dict[label] = [example]
    else:
        rel_dict[label].append(example)
print("Done building dictionary")  
    
# example for each relation
for rel in rel_dict.keys():
    ex = rel_dict[rel][0]
    print(rel,ex.entity_1,ex.entity_2)


# In[6]:


# how to reconstruct full context

# ex = train_data[0]
# print(ex)
# print("\n full context:")
# s = ex.snippet[0]
# print(' '.join((s.left, s.mention_1, s.middle, s.mention_2, s.right)))


# In[7]:


# def rebuild_text(ex):
#     rebuilt_ex = []
#     for s in ex.snippet:
#         text = ' '.join((s.left, s.mention_1, s.middle, s.mention_2, s.right))
#         rebuilt_ex.append(text)
#     return rebuilt_ex


# In[8]:


# def build_text_from_snippet(s):
#     text = ' '.join((s.left, s.mention_1, s.middle, s.mention_2, s.right))
#     return text


# In[9]:


# def rebuild_corpus(data):
#     corpus = []
#     for ex in data:
#         corpus.append(rebuild_text(ex)) 
#     return corpus


# In[10]:


# def extract_key_sents(data):
#     key_sents = []
#     for ex in data:
#         m1 = ex.snippet[0].mention_1
#         m2 = ex.snippet[0].mention_2
#         text = build_text_from_snippet(ex.snippet[0])
#         doc = nlp(text)
#         for sent in doc.sents:
# #             print(sent)
#             if m1 in sent.string and m2 in sent.string:
#                 key_sents.append(sent)
#                 continue
                
#     return key_sents


# In[11]:


# key_sents = extract_key_sents(train_data[:100])
# print(type(key_sents[0]))
# for sent in key_sents:
#     for chunk in sent.noun_chunks:
#         print(chunk.label_, chunk.text, chunk.root.text, chunk.root.dep_,
#           chunk.root.head.text)
#     for token in sent: 
#         print(token.text, token.dep_, token.head.text, token.head.pos_,
#               [child for child in token.children])


# In[ ]:





# In[12]:


def tag_tokens(doc):
    tagged_ex = []
    
    for w in doc:
        if w.orth_ == "MENTION_1" or w.orth_ == "MENTION_2":
            tagged_ex.append(w.orth_)
        else:
            tagged_ex.append(w.pos_)
            
    tagged_ex = " ".join(tagged_ex)
    
    return tagged_ex


# In[13]:


def lemmatize(doc):
    lemmas = []
    
    for w in doc:
        if w.lemma_ == "-PRON-" or w.orth_ == "MENTION_1" or w.orth_ == "MENTION_2":
            lemmas.append(w.orth_)
        else:
            lemmas.append(w.lemma_)
    
    lemmas = " ".join(lemmas)
    
    return lemmas


# In[14]:


##################################################################################################
# 2.1 PERFORM NLP ON CORPUS DATA
##################################################################################################

def perform_nlp(data, verbose=True):
    
    if verbose:
        print("{} instances in data".format(len(data)))
        print("first instance looks like {}".format(data[0]))
        
    c = 0
    docs = []
    for instance in data:
        instance_context = []
        for s in instance.snippet:
            context = nlp(s.left + " MENTION_1 " + s.middle + " MENTION_2 " + s.right)
            instance_context.append(context)
        docs.append(instance_context)
        c += 1
    
        if verbose:
            if c % 1000 == 0:
                print("{} instances processed.".format(c))
        
    if verbose:
        print(len(docs))
        print(docs[0])
        print("Structure of context data is: {}-{}-{}".format(type(docs),
                                                              type(docs[0]),
                                                              type(docs[0][0])
                                                             )
             )
    
    return docs


# In[15]:


##################################################################################################
# 2. EXTRACT FEATURES and BUILD CLASSIFIER
##################################################################################################

# Turn data into numerical features

def SelectContext(data, verbose=True):
    """BOW feature extraction"""
    only_context_data = []
    for instance in data:
        instance_context = []
        for s in instance.snippet:
            context = s.left + " MENTION_1 " + s.middle + " MENTION_2 " + s.right
            instance_context.append(context)
        only_context_data.append(' '.join(instance_context))
    if verbose:
        print(len(data))
        print(len(only_context_data))
        print(data[0])
        print(only_context_data[0])
    return only_context_data


# In[16]:


# Extract two simple features
def ExractSimpleFeatures(data, verbose=True):
    featurized_data = []
    for instance in data:
        featurized_instance = {'mid_words':'', 'distance':np.inf}
        for s in instance.snippet:
            if len(s.middle.split()) < featurized_instance['distance']:
                featurized_instance['mid_words'] = s.middle
                featurized_instance['distance'] = len(s.middle.split())
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(data))
        print(len(featurized_data))
        print(data[0])
        print(featurized_data[0])
        print(featurized_data[1])
    return featurized_data


# In[17]:


# def SelectLemmatizedContext(data, verbose=True):
    
#     processed_data = perform_nlp(data)
    
#     lemmatized_data = []
#     for processed_instance in processed_data:
#         instance_lemmas = []
#         for doc in processed_instance:  
#             lemmas = lemmatize(doc)
#             instance_lemmas.append(lemmas)
#         lemmatized_data.append(' '.join(instance_lemmas))
#     if verbose:
#         print(len(processed_data))
#         print(len(lemmatized_data))
#         print(processed_data[0])
#         print(lemmatized_data[0])
#     return lemmatized_data


# In[18]:


# def SelectTaggedContext(data, verbose=True):
    
#     processed_data = perform_nlp(data)
    
#     tagged_data = []
#     for processed_instance in processed_data:
#         instance_tags = []
#         for doc in processed_instance:  
#             tags = tag_tokens(doc)
#             instance_tags.append(tags)
#         tagged_data.append(' '.join(instance_tags))
#     if verbose:
#         print(len(processed_data))
#         print(len(tagged_data))
#         print(processed_data[0])
#         print(tagged_data[0])
#     return tagged_data


# In[19]:


# Inspect feature extractions
# context_simple = SelectContext(train_data)
# context_lemmas = SelectLemmatizedContext(processed_train_data)
# context_tagged = SelectTaggedContext(processed_train_data)


# In[20]:


class SimpleFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return ExractSimpleFeatures(X, verbose=False)


# In[21]:


class BowFeaturizer(BaseEstimator, TransformerMixin):
    """Extract features from each isntance for DictVectorizer"""
    def __init__(self, *featurizers):
        self.featurizers = featurizers
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return SelectContext(X, verbose=False)
    


# In[22]:


union = FeatureUnion([("simple", SimpleFeaturizer(ExractSimpleFeatures)), ("bow", BowFeaturizer(SelectContext))])


# In[ ]:





# In[ ]:





# In[23]:


# Transform dataset to features
# train_data_featurized = ExractSimpleFeatures(train_data, verbose=False)

# Transform dataset to features
# train_data_featurized = SelectContext(train_data, verbose=False)

# print(train_data_featurized[:3])

# union = FeatureUnion([("simple", DictVectorizer()), ("bow", CountVectorizer())])

# Transform labels to numeric values
le = LabelEncoder()
train_labels_featurized = le.fit_transform(train_labels)

# print(train_labels_featurized.shape)

# clf = Pipeline([
#     ('cv', CountVectorizer(ngram_range=(2, 3))), # creates normal bow model
#     ('tfidf', TfidfTransformer()), # passes bow model and transforms to tfidf
#     ('logit', LogisticRegression()), # passes transformer to LR
# ])

# clf = make_pipeline(CountVectorizer(ngram_range=(2, 3)), TfidfTransformer(), LogisticRegression())

# Fit model one vs rest logistic regression    
# clf = make_pipeline(DictVectorizer(), LogisticRegression())

clf = make_pipeline(union, LogisticRegression())


# In[24]:


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


# In[ ]:


evaluateCV(clf, le, train_data, train_labels_featurized)


# In[ ]:


# A check for the average F1 score

f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')

def evaluateCV_check(classifier, X, y, verbose=True):
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0) 
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring = f_scorer)
    print("\nCross-validation scores (StratifiedKFold): ", scores)
    print("Mean cv score (StratifiedKFold): ", scores.mean())


# In[ ]:


evaluateCV_check(clf, train_data_featurized, train_labels_featurized)


# In[ ]:


##################################################################################################
# 4. TEST PREDICTIONS and ANALYSIS
##################################################################################################

# Fit final model on the full train data
clf.fit(train_data_featurized, train_labels_featurized)

# Predict on test set
test_data, test_labels = load_data('../data/test-covered.json.txt', verbose=False)
print(len(test_labels))
test_data_featurized = SelectContext(test_data, verbose=False)
test_label_predicted = clf.predict(test_data_featurized)
print(len(test_label_predicted))
# Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
print(len(test_label_predicted_decoded))
print(test_label_predicted_decoded[:2])
f = open("outputs/test_labels.txt", 'w', encoding="utf-8")
for label in test_label_predicted_decoded:
    f.write(label+'\n')


# In[ ]:


# Feature analisys - print N most informative
# !! Make changes in this function when you change the pipleine!!
def printNMostInformative(classifier,label_encoder,N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = classifier.named_steps['countvectorizer'].get_feature_names()

    coef = classifier.named_steps['logisticregression'].coef_    
    print(coef.shape)
    for rel in label_encoder.classes_:
        rel_id = label_encoder.transform([rel])[0]
        coef_rel = coef[rel_id]
        coefs_with_fns = sorted(zip(coef_rel, feature_names))
        top_features = coefs_with_fns[-N:]
        print("\nClass {} best: ".format(rel))
        for feat in top_features:
            print(feat)        
        
print("Top features used to predict: ")
# show the top features
printNMostInformative(clf,le,2)


# In[ ]:





# In[ ]:




