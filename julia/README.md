# Evaluation: Julia's models

Here I will post the scores for my latest models, so you can see what I try and what results I get.

| **model**  | **score** Mean cv score (StratifiedKFold)| **comments** |
| -------- | -------- | -------- |
| BASELINE  | 0.7745792487781428 | |
| + two target entities | 0.7890200518748698 | (new baseline) |
| without stop words  | 0.782739810180745 | |
| TFiDF  | 0.7518001918286948 | |
| **ngram_range=(1, 3)** | **0.8100545275185225** | the best results so far |
| max_df=0.7  | 0.7801255472002808 | |
| word2vec (trained on our data) | 0.7004649820878264 | only entities are vectorized; tested on train data only |
| pretrained word2vec | 0.5483315593034295 | context words vectorized (without entities; see more below) |

## 1. Embeddings
When word embeddings are used, there are two vectors for each snippet (for entities). Here we should decide how to concatenate them: 1) can be multiplied; 2) similarity (I do not think it helps in this case); 3) ...
Another question how to tie all snippets within one instance.

Other difficulties with word embeddings:
* some mentions (target entities) are the phases like *"book of the same name"*. I am not sure that the averaged vector for this phrase will give what we are looking for.
* many proper names, which are OOV -> problematic for test data (that is why it is evaluated only on train data so far).

**New results with vectorized context skipping entities:** model = api.load(*"glove-wiki-gigaword-100"*) was used. The results are not promising so far, but I vectorised all tokens from the context (including punctuation), so this part can be still improved: for example by vectorizing just content words.
only 0.5483315593034295 with CV on all train data.


## 2. Overfitting using entities
To check this issue, I evaluated the results on the development dataset. (To get dev data, I had to split the train data: 0.8-train, 0.2-dev; the provided test are not labeled). With CountVectorizer(1, 3) I did not notice any sufficient difference between train and test evaluation:

|  | with entities | without entities |
| -------- | -------- | -------- |
| CountVectorizer(ngram_range=(1, 1)) | 0.7885812878907923 | 0.7693284957354679 |
|  | 0.7859857572352148 | 0.7667723850710316 |
| CountVectorizer(ngram_range=(1, 3)) | TRAIN DATA: 0.7963197900680725 | TRAIN DATA: 0.788512940884961 |
|  | DEV DATA: 0.7930781252127049 | DEV DATA: 0.7930781252127049 |


---------------
.ipython and .py are not identical. Development set is used in .ipython. Embeddings are in .py now.
