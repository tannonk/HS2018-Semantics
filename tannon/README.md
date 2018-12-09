## Evaluation: Tannon's models
## 
Here I will post the scores for my latest models, so you can see what I
try and what results I get.

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
| splitting data + ngram range(1, 3) + EntityLength + SimpleFeats + SyntacticFeats | 0.8721445323326972 |  |
| splitting data + ngram range(1, 3) + EntityLength + SimpleFeats | 0.869709024613503 |  |


## Approach:

Load data, splitting snippets as new instances. Results in ~ 43320 samples
Features:
BOW with ngrams: CountVectorizer(ngram_range(1, 3))
Entity length: number of words in each entity and their concatenation
Simple features: considers aspects of middle segments (adapted from PA4\_explained)
Dependency length: length of shortest path between mentions 

## Problems encountered:
NER Labelling: J&M suggest using named entity types and their
concetenation as features. So far, I haven't been able to reliably extract these labels.