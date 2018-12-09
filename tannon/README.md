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
| ngram_range(1, 3) + EntityLength + SimpleFeats | 0.8549579216500934 |  |

 
## Problems encountered:
NER Labelling: J&M suggest using named entity types and their
concetenation as features. So far, I haven't been able to reliably extract these labels.