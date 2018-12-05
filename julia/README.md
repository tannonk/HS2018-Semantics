## Evaluation: Julia's models

Here I will post the scores for my latest models, so you can see what I try and what results I get.

| *model*  | *score* Mean cv score (StratifiedKFold)| *comments* |
| -------- | -------- | -------- |
| BASELINE  | 0.7745792487781428 | |
| + two target entities | 0.7890200518748698 | (new baseline) |
| without stop words  | 0.782739810180745 | |
| TFiDF  | 0.7518001918286948 | |
| *ngram_range=(1, 3)* | *0.8100545275185225* | |
| max_df=0.7  | 0.7801255472002808 | the best results so far|
| word2vec | 0.7004649820878264 | (tested on train data only!!!) |
|   |   |  |

When word embeddings are used, there are two vectors for each snippet (for entities). Here we should decide how to concatenate them: 1) can be multiplied; 2) similarity (I do not think it helps in this case); 3) ...
Another question how to tie all snippets within one instance.

Other difficulties with word embeddings:
* some mentions (target entities) are the phases like **"book of the same name"**. I am not sure that the averaged vector for this phrase will give what we are looking for.
* many proper names, which are OOV -> problematic for test data (that is why it is evaluated only on train data so far).

---------------
.ipython and .py are not identical at the moment. Word embeddings are in .ipython.
