# !/usr/bin/env python3
# -*- coding: utf8 -*-

import json
from collections import Counter, defaultdict, namedtuple


infile = "../data/test.json.txt"

PairExample = namedtuple('PairExample',
    'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
    'left, mention_1, middle, mention_2, right, direction')

def load_test_data(file, verbose=True):
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

    return data, labels

def check_rels(test_data, test_labels, rel):
    unique_middle_segments = set()
    rel_count = 0

    for d, l in zip(test_data, test_labels):
        if l == rel:
            rel_count += 1
            unique_middle_segments.add(d.snippet[0].middle)

    print("There are {} unique middle segments for the relation {} from {} samples.".format(len(unique_middle_segments), rel, rel_count))


test_data, test_labels = load_test_data(infile, verbose=False)
rels = ["author", "worked_at", "has_spouse", "capital"]

for rel in rels:
    check_rels(test_data, test_labels, rel)
