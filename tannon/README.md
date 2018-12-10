## Evaluation: Tannon's models
## 
Scores for tested models.

| **model**  | **score** Mean cv score (StratifiedKFold)| **comments** |
| -------- | -------- | -------- |
| BASELINE  | 0.7745792487781428 |  |
| +tfidfvectorizor | 0.750413578679886 | |
| lemmatized  | 0.7664650091110985  |  |
| BASELINE + EntityLength | 0.7818803012927732 | unions features |
| BASELINE + EntityLength + SimpleFeats | 0.8108522380368186 |  |
| BASELINE + EntityLength + SimpleFeats | 0.8279038078401164 |  |
| ngram range(1, 3) + EntityLength + SimpleFeats | 0.8549579216500934 |  |
| splitting data + ngram range(1, 3) + EntityLength + SimpleFeats | 0.8716350652593686 |  |
| splitting data + ngram range(1, 3) + EntityLength + SimpleFeats + SyntacticFeats | 0.8721445323326972 | best result but very time consuming |
| splitting data + ngram range(1, 3) + EntityLength + SimpleFeats | 0.869709024613503 | final result 10/12/18 |


## Approach:

Load data, splitting snippets as new instances. Results in ~ 43320 samples
Features:
BOW with ngrams: CountVectorizer(ngram_range(1, 3))
Entity length: number of words in each entity and their concatenation
Simple features: considers aspects of middle segments and direction (adapted from PA4\_explained)
<!-- Dependency length: length of shortest path between mentions  -->

## Problems encountered:
NER Labelling: J&M suggest using named entity types and their
concetenation as features. So far, I haven't been able to reliably extract these labels.

## Syntactic features experiment
Using Spacy's pipeline, we labeled data instances with pos tags and analyzed them for dependency relations.
A separate feature, which considered pos tags of words surrounding each mention and the length of the dependency path between the two mentions, was used in testing. However, this feature extended the training time drastically and improved cross validation scores by only 0.2%.

## Features explained

Context: BOW ngram model
Entity Length: considers number of words in entity1, entity2 and their concatenated length
Simple Features: considers words between mentions, distance between mentions and direction (i.e. fwd or bwd)


