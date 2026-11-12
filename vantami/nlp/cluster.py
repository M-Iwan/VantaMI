from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim


def lda_cluster(tokenized_documents, no_below: int = 3, no_above: float = 0.7, num_topics: int = 64,
                chunksize: int = 1024, passes: int = 16, random_state: int = 42):
    """
    Train a Latent Dirichlet Allocation model on tokenized documents.

    Parameters
    ----------
    tokenized_documents: list[list[str]]
        Tokenized documents used to build the dictionary and corpus.
    no_below: int, default=3
        Minimum number of documents in which a token must appear.
    no_above: float, default=0.7
        Maximum proportion of documents in which a token may appear.
    num_topics: int, default=64
        Number of topics to generate.
    chunksize: int, default=1024
        Number of documents processed in each training chunk.
    passes: int, default=16
        Number of passes over the corpus during training.
    random_state: int, default=42
        Seed used to make model training reproducible.

    Returns
    -------
    results: dict
        Dictionary containing the dictionary, corpus, trained LDA model, coherence
        model and score, extracted topics, and pyLDAvis visualization.
    """
    dictionary = corpora.Dictionary(tokenized_documents)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    corpus = [dictionary.doc2bow(document) for document in tokenized_documents]

    lda_model = LdaModel(corpus = corpus, id2word=dictionary, num_topics=num_topics, chunksize=chunksize,
                         passes=passes, alpha="auto", per_word_topics=True, random_state=random_state)

    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence="c_v")
    coherence_score = coherence_model.get_coherence()

    topics = lda_model.print_topics(num_topics=num_topics, num_words=16)
    lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)

    results = {
        "dictionary": dictionary,
        "corpus": corpus,
        "lda_model": lda_model,
        "coherence_model": coherence_model,
        "coherence_score": coherence_score,
        "topics": topics,
        "display": lda_display
    }

    return results