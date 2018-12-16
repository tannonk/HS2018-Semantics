# !/usr/bin/env python3
# -*- coding: utf8 -*-

import json
from collections import Counter, defaultdict, namedtuple


infile = "data/test.json.txt"

PairExample = namedtuple('PairExample',
    'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
    'left, mention_1, middle, mention_2, right, direction')


def load_test_data(file, verbose=True):
    f = open(file, 'r', encoding='utf-8')
    data = []
    labels = []
    for i, line in enumerate(f):
        instance = json.loads(line)
        if i == 0:
            if verbose:
                print('json example:')
                print(instance)
        #'relation, entity_1, entity_2, snippet' fileds for each example
        #'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'], instance['entity_2'], [])
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


def get_context(data, embed_mode=False):
    all_data = []
    for instance in data:
        s_context = []
        for s in instance.snippet:
            if embed_mode:
                # s_context.append(' '.join((s.left, s.mention_1.replace(" ", "_"),
                #                            s.middle, s.mention_2.replace(" ", "_"),
                #                            s.right)))
                s_context.append(' '.join(s.middle))
                # s_context.append(' '.join((s.left, s.middle, s.right)))

            else:
                # s_context.append(' '.join((s.left, s.mention_1, s.middle, s.mention_2, s.right)))
                s_context.append(' '.join((s.left, "ENTITY_1", s.middle, "ENTITY_2", s.right)))
        all_data.append(' '.join(s_context))

    # print(len(all_data))
    return all_data


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


def print_errors(test_data, test_labels, rel):
    with open("test_labels.txt", "r") as pred_file:
        test_predicted = pred_file.read().strip().split("\n")
        print(len(test_data))
        print(len(test_labels))
        print(len(test_predicted))
        test_data_text = get_context(test_data)
        n_errors = 0
        n_error_classes = {}
        for instance, true, pred in zip(test_data_text, test_labels, test_predicted):
            # most of the errors are TN, that is why this case is printed out here:
            if true == rel and pred != rel:
                print(instance, true, pred)
                n_errors += 1
                print("\n")
                if pred in n_error_classes:
                    n_error_classes[pred] += 1
                else:
                    n_error_classes[pred] = 1
        print("Relation: {}".format(rel))
        print("Number of TN errors: ", n_errors)
        for wrong_class in n_error_classes:
            print("--> classified as {}: {} times.".format(wrong_class,
                                                           n_error_classes[wrong_class]))


for rel in rels:
    check_rels(test_data, test_labels, rel)

print_errors(test_data, test_labels, rel="worked_at")
