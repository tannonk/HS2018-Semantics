                          Precision   Recall    F-score


Baseline: clf = make_pipeline(CountVectorizer(), LogisticRegression())

    macro-average			0.789      0.736      0.775


clf = make_pipeline(CountVectorizer(), TfidfTransformer(), LogisticRegression())

    macro-average			0.792      0.676      0.750   


clf = make_pipeline(CountVectorizer(), TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True), LogisticRegression())

    macro-average			0.792      0.676      0.750 


clf = make_pipeline(CountVectorizer(ngram_range=(1,3)), TfidfTransformer(), LogisticRegression())

    macro-average			0.779      0.584      0.673 


clf = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression())

    macro-average			0.821      0.742      0.797


CountVectorizer(ngram_range=(1,3)), LogisticRegression(), mentions replaced with placeholders, snippets form separate samples

    macro-average			0.878      0.805      0.850



 



