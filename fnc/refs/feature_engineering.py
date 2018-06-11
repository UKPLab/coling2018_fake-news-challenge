import math
import os
import os.path as path
from datetime import datetime
import nltk
import numpy as np
import regex as re
import pickle
import retinasdk
import string
import pandas as pd
import csv
import collections
from time import time
from nltk.corpus import reuters, stopwords
from collections import defaultdict, Counter
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import LatentDirichletAllocation, NMF

from tqdm import tqdm
from itertools import zip_longest
from fnc.refs.feature_engineering_helper import word_ngrams
from fnc.refs.feature_engineering_helper import topic_models
from fnc.utils.doc2vec import avg_embedding_similarity
from fnc.utils.loadEmbeddings import LoadEmbeddings
from fnc.utils.stanford_parser import StanfordMethods
from fnc.utils.tf_idf_helpers import tf_idf_helpers
from fnc.utils.hungarian_alignment import hungarian_alignment_calculator
from fnc.utils.data_helpers import sent2stokens_wostop, text2sent, get_tokenized_lemmas
from fnc.utils.word_mover_distance import computeAverageWMD
from fnc.settings import myConstants
from fnc.refs.utils.generate_test_splits import kfold_split
import fnc.refs.feature_engineering_helper.readability_indices as fe_util

_wnl = nltk.WordNetLemmatizer()

"""
This file is based on the fnc-1-baseline implementation https://github.com/FakeNewsChallenge/fnc-1-baseline
"""

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def clear_unwanted_chars(mystring):
    return str(mystring.encode('latin', errors='ignore').decode('latin'))


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file, bodyId, feature, headId="", fold=""):
    if not os.path.isfile(feature_file):
        if 'stanford' in feature:
            feats = feat_fn(headlines, bodies, bodyId, headId)
        elif 'single_flat_LSTM_50d_100' in feature:
            feats = feat_fn(headlines, bodies, fold)

        else:
            feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)

def gen_non_bleeding_feats(feat_fn, headlines, bodies, headlines_test, bodies_test, features_dir, feature,
                           fold):
    """
    Similar to gen_or_load_feats() it generates the non bleeding features and save them on the disk
    """
    feature_file = "%s/%s.%s.npy" % (features_dir, feature, fold)
    if not os.path.isfile(feature_file):
        print (str(datetime.now()) + ": Generating features for: " + feature + ", fold/holdout: " + str(fold))

        X_train, X_test = feat_fn(headlines, bodies, headlines_test, bodies_test)

        if (str(fold) != 'holdout'):
            np.save("%s/%s.%s.npy" % (features_dir, feature, fold), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, fold), X_test)
        else:
            np.save("%s/%s.%s.npy" % (features_dir, feature, 'holdout'), X_train)
            np.save("%s/%s.%s.test.npy" % (features_dir, feature, 'holdout'), X_test)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X

def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))
    return X

###########################
# NEW FEATURES START HERE #
###########################
def NMF_cos_50(headlines, bodies):
    """
    Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=50, include_holdout=False, include_unlbled_test=False)

def NMF_cos_300(headlines, bodies):
    """
    Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=False, include_unlbled_test=False)

def NMF_cos_300_holdout_unlbled_test(headlines, bodies):
    """
    Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=True)

def latent_dirichlet_allocation_25(headlines, bodies):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test+holdout+ unlabeled test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """
    return topic_models.latent_dirichlet_allocation_cos(headlines, bodies, n_topics=25, include_holdout=False,
                                                        use_idf=False, term_freq=True, incl_unlbled_test=False)

def latent_dirichlet_allocation_300(headlines, bodies):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test+holdout+ unlabeled test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """
    return topic_models.latent_dirichlet_allocation_cos(headlines, bodies, n_topics=300, include_holdout=False,
                                                        use_idf=False, term_freq=True, incl_unlbled_test=False)

def latent_dirichlet_allocation_25_holdout_unlbled_test(headlines, bodies):
    """
    Sklearn LDA implementation based on the 5000 most important words (based on train+test+holdout+ unlabeled test data's term freq => bleeding).
    Returns feature vector of cosinus distances between the topic models of headline and bodies.

    Links:
        https://pypi.python.org/pypi/lda, bottom see suggestions like MALLET, hca
        https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
        https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    """
    return topic_models.latent_dirichlet_allocation_cos(headlines, bodies, n_topics=25, include_holdout=True,
                                                        use_idf=False, term_freq=True, incl_unlbled_test=True)

def latent_semantic_indexing_gensim_300_concat_holdout(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim_300_concat_OLD are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    return topic_models.latent_semantic_indexing_gensim_concat(headlines, bodies, n_topics=300, include_holdout=True,
                                                               include_unlbled_test=False)

def latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim_300_concat_OLD are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    return topic_models.latent_semantic_indexing_gensim_concat(headlines, bodies, n_topics=300, include_holdout=True,
                                                               include_unlbled_test=True)

def NMF_concat_300_holdout(headlines, bodies):
    """
    Implements non negative matrix factorization. Concatenates the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=False,
                                   cosinus_dist=False)

def NMF_concat_300_holdout_unlbled_test(headlines, bodies):
    """
    Implements non negative matrix factorization. Concatenates the resulting head and body vector.
    """
    return topic_models.NMF_topics(headlines, bodies, n_topics=300, include_holdout=True, include_unlbled_test=True,
                                   cosinus_dist=False)

def word_unigrams_5000_concat_tf_l2_holdout(headlines, bodies):
    """
    Simple bag of words feature extraction with term freq of words as feature vectors, length 5000 head + 5000 body,
    concatenation of head and body, l2 norm and bleeding (BoW = train+test+holdout+unlabeled test set).
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    # get headlines and bodies of train, test and holdout set
    h, b = word_ngrams.get_head_body_tuples(include_holdout=True)

    # create the vocab out of the BoW
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=False,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X

def word_unigrams_5000_concat_tf_l2_holdout_unlbled_test(headlines, bodies):
    """
    Simple bag of words feature extraction with term freq of words as feature vectors, length 5000 head + 5000 body,
    concatenation of head and body, l2 norm and bleeding (BoW = train+test+holdout+unlabeled test set).
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    # get headlines and bodies of train, test and holdout set
    h, b = word_ngrams.get_head_body_tuples(include_holdout=True)


    # Comment out for clean ablation tests
    # add the unlabeled test data words to the BoW of test+train+holdout data
    h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_unlbled_test()
    h.extend(h_unlbled_test)
    b.extend(b_unlbled_test)

    # create the vocab out of the BoW
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h, b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X



#'PPDB' paraphrase database
#hungarian alignment score
#computing score of each word of headline with each word of body - very resource-hungry
def ppdb(headlines, bodies):
    myHungarian_calculator = hungarian_alignment_calculator()
    x = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(myHungarian_calculator.calc_hungarian_alignment_score(headline, body))
    return x


def hedging_features(headlines, bodies):
    _hedging_seed_words = \
        [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed']
    # Taken from https://github.com/wooorm/hedges/blob/master/index.json
    _hedging_seed_words = \
        [
            "a bit",
            "about",
            "actually",
            "allege",
            "alleged",
            "almost",
            "almost never",
            "always",
            "and all that",
            "and so forth",
            "apparent",
            "apparently",
            "appear",
            "appear to be",
            "appeared",
            "appears",
            "approximately",
            "around",
            "assume",
            "assumed",
            "assumes",
            "assumption",
            "at least",
            "basically",
            "be sure",
            "believe",
            "believed",
            "believes",
            "bunch",
            "can",
            "certain",
            "certainly",
            "clear",
            "clearly",
            "conceivably",
            "consider",
            "considered",
            "considers",
            "consistent with",
            "could",
            "couple",
            "definite",
            "definitely",
            "diagnostic",
            "don't know",
            "doubt",
            "doubtful",
            "effectively",
            "estimate",
            "estimated",
            "estimates",
            "et cetera",
            "evidently",
            "fairly",
            "few",
            "find",
            "finds",
            "found",
            "frequently",
            "generally",
            "guess",
            "guessed",
            "guesses",
            "hopefully",
            "if i'm understanding you correctly",
            "improbable",
            "in general",
            "in my mind",
            "in my opinion",
            "in my understanding",
            "in my view",
            "inconclusive",
            "indicate",
            "kind of",
            "largely",
            "like",
            "likely",
            "little",
            "look like",
            "looks like",
            "mainly",
            "many",
            "may",
            "maybe",
            "might",
            "more or less",
            "most",
            "mostly",
            "much",
            "must",
            "my impression",
            "my thinking is",
            "my understanding is",
            "necessarily",
            "occasionally",
            "often",
            "overall",
            "partially",
            "perhaps",
            "possibility",
            "possible",
            "possibly",
            "practically",
            "presumable",
            "presumably",
            "pretty",
            "probability",
            "probable",
            "probably",
            "quite",
            "quite clearly",
            "rare",
            "rarely",
            "rather",
            "read",
            "really",
            "roughly",
            "say",
            "says",
            "seem",
            "seemed",
            "seems",
            "seldom",
            "several",
            "should",
            "so far",
            "some",
            "somebody",
            "somehow",
            "someone",
            "something",
            "something or other",
            "sometimes",
            "somewhat",
            "somewhere",
            "sort of",
            "speculate",
            "speculated",
            "speculates",
            "suggest",
            "suggested",
            "suggestive",
            "suggests",
            "suppose",
            "supposed",
            "supposedly",
            "supposes",
            "surely",
            "tend",
            "their impression",
            "think",
            "thinks",
            "thought",
            "understand",
            "understands",
            "understood",
            "unlikely",
            "unsure",
            "usually",
            "virtually",
            "will",
            "would"
        ]

    def calculate_hedging_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _hedging_seed_words for t in tokens]) % 2
    def contains_hedging_seeed(text):
        tokens = get_tokenized_lemmas(text)
        return (min(1,sum([t in _hedging_seed_words for t in tokens])))
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        #features.append(calculate_hedging_polarity(clean_headline))
        #features.append(calculate_hedging_polarity(clean_body))
        features.append(contains_hedging_seeed(clean_headline))
        features.append(contains_hedging_seeed(clean_body))
        X.append(features)
    return np.array(X)

# End Features taken from: https://github.com/willferreira/mscproject
##############################################################

def load_embeddings(headlines, bodies):
    # embedding parameters:
    embedding_size = 300
    vocab_size = 3000000
    embeddPath = "%s/data/embeddings/google_news/GoogleNews-vectors-negative300.bin.gz" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    embeddData = path.normpath("%s/data/" % (path.dirname(path.abspath(embeddPath))))
    binary_val = True
    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size, embedding_size=embedding_size, binary_val=binary_val)
    #     print('Loaded embeddings: Vocab-Size: ' + str(vocab_size) + ' \n Embedding size: ' + str(embedding_size))
    return embedding_size, embeddings



# calculate average sentence vector and compare headline with whole body text
# same like avg_embed in model.py
def sen2Doc_headline_wholebody(headlines, bodies):

    def headline_wholebody(embeddings, headline, body):
        headline_w = sent2stokens_wostop(headline)
        body_w = sent2stokens_wostop(body)
        sim = avg_embedding_similarity(embeddings, embedding_size, ' '.join(headline_w), ' '.join(body_w))
        features = []
        features.append(sim)
        return features

    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(headline_wholebody(embeddings, headline, body))
    return x

# calculate average sentence vector and compare headline with each sentence, use highest similarity
def sen2sen_similarity_max(headlines, bodies):

    def similarity(embeddings, headline, body):
        sentence_list = []
        score = 0
        sentence_list = text2sent(body)
        for sentence in sentence_list:
            # compare both sentences - vectors not necessary, since this procedure works with text
            # note: avg_embeddings_similarity tokenizes and lemmatizes the sentences prior to calculation, so no pre-assessment is necessary (Sentence to tokens without stopwords)
            temp_score = avg_embedding_similarity(embeddings, embedding_size, headline, sentence)
            # store the highest similarity score
            score=max(score, temp_score)

        features = []
        features.append(score)
        return features

    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(similarity(embeddings, headline, body))
    return x

# calculate word_mover_distance from headline to each sentence, use lowest distance
def word_mover_distance_similarity_sentence_min(headlines, bodies):

    def similarity(embeddings, headline, body):
        distance = 99999
        sentence_list = []
        sentence_list = text2sent(body)
        embedding_size, embeddings = load_embeddings(headline, body)
        for sentence in sentence_list:
            temp_distance = abs(computeAverageWMD(embeddings, headline, sentence))
            # store the lowest distance
            # Note: Distance is not normallized!!
            distance=min(distance, temp_distance)

        features = []
        features.append(distance)
        return features
    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(similarity(embeddings, headline, body))
    return x

# calculate word_mover_distance from headline to whole body text
def word_mover_distance_wholebody(headlines, bodies):
    def similarity(embeddings, headline, body):
        embedding_size, embeddings = load_embeddings(headline, body)
        distance = abs(computeAverageWMD(embeddings, headline, body))
        features = []
        features.append(distance)
        return features
    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(similarity(embeddings, headline, body))
    return x

# compare sdm of the headline with the sdm of the whole body
def sdm_sim(headlines, bodies):
    def similarity(headline, body):
        clean_headline = clean(headline)
        clean_body = clean(body)
        fullClient = retinasdk.FullClient("e8bf8de0-fe52-11e6-b22d-93a4ae922ff1", apiServer="http://api.cortical.io/rest", retinaName="en_associative")

        RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
        clean_body = RE.sub(u'', clean_body)
        #         clean_body = clean_body.encode('ascii', 'ignore')
        clean_body = clean_body.encode('utf8', 'ignore')
        clean_body = clean_body.decode('utf8', 'ignore')
        #         print(clean_body)
        clean_body.replace("0x6e", " ")
        #         newdata = clean_body[:start] + clean_body[end:]
        #         clean_body = clean_body.translate(None, '0x6e')
        comp_with_stop_words = fullClient.compare('[{"text": "'+clean_headline+'"}, {"text": "'+clean_body +'"}]')
        sim = comp_with_stop_words.cosineSimilarity

        features = []
        features.append(sim)
        return features
    x = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        x.append(similarity(headline, body))
    return x


def stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    myStanfordmethods = StanfordMethods()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_word_sim(embeddings, headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id,clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers)
        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo('headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo('body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        try:
            noun_sim = avg_embedding_similarity(embeddings, embedding_size, ' '.join(headline_nouns), ' '.join(body_nouns))
        except Exception as e:
            #print(e)
            #print('Problem with nouns for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            #print(body_nouns)
            noun_sim = -1
        if math.isnan(noun_sim):
            #print('NAN for nouns for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline) + '\n \n Body-text: ' + str(ranked_sentences) + ' \n \n Body-verbs: ' + str(body_verbs) + '\n \n Headline-verbs:' + str(headline_verbs))
            #print(body_nouns)
            noun_sim = -1

        try:
            verb_sim = avg_embedding_similarity(embeddings, embedding_size, ' '.join(headline_verbs), ' '.join(body_verbs))
        except Exception as e:
            #print(e)
            #print('Problem with verbs for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            #print(body_verbs)
            verb_sim = -1

        if math.isnan(verb_sim):
            #print('NAN for verbs for dataset with headline for body ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline) + '\n \n Body-text: ' + str(ranked_sentences) + ' \n \n Body-verbs: ' + str(body_verbs) + '\n \n Headline-verbs:' + str(headline_verbs))
            #print(body_verbs)
            verb_sim = -1

        features = []
        features.append(noun_sim)
        features.append(verb_sim)

        return features

    x = []
    embedding_size, embeddings = load_embeddings(headlines, bodies)
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_word_sim(embeddings, headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x


def stanford_based_verb_noun_sim_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_based_verb_noun_sim_2sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=2):
    return stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_based_verb_noun_sim_3sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=3):
    return stanford_based_verb_noun_sim(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)


def stanford_ppdb_score(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    myStanfordmethods = StanfordMethods()
    myHungarian_calculator = hungarian_alignment_calculator()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_ppdb_score(headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id,clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers)

        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo('headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo('body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        try:
            noun_ppdb_score = myHungarian_calculator.calc_hungarian_alignment_score(' '.join(headline_nouns), ' '.join(body_nouns))
        except Exception as e:
            #print(e)
            #print('Problem with ppdb score in nouns for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            #print(body_nouns)
            noun_ppdb_score = -99

        try:
            verb_ppdb_score = myHungarian_calculator.calc_hungarian_alignment_score( ' '.join(headline_verbs), ' '.join(body_verbs))
        except Exception as e:
            #print(e)
            #print('Problem with ppdb score in verbs for dataset with headline ID: ' + str(body_id) + '\n Headline-text: ' + str(clean_headline))
            #print(body_verbs)
            verb_ppdb_score = -99

        features = []
        features.append(noun_ppdb_score)
        features.append(verb_ppdb_score)

        return features

    x = []
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_ppdb_score(headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x


def stanford_ppdb_score_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_ppdb_score(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_ppdb_score_2sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=2):
    return stanford_ppdb_score(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_ppdb_score_3sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=3):
    return stanford_ppdb_score(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)


def stanford_sentiment(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    myStanfordmethods = StanfordMethods()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_sentiment(headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id,clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers)

        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo('headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo('body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        #get average sentiment of sentences
        head_sentiment_avg = float(sum(head_sentiment))/len(head_sentiment)
        body_sentiment_avg = float(sum(body_sentiment))/len(body_sentiment)

        features = []
        features.append(head_sentiment_avg)
        features.append(body_sentiment_avg)

        return features

    x = []
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_sentiment(headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x

def stanford_sentiment_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_sentiment(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_sentiment_2sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=2):
    return stanford_sentiment(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_sentiment_3sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=3):
    return stanford_sentiment(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)


def stanford_negation_features(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    myStanfordmethods = StanfordMethods()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_negation(headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id,clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers)

        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo('headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo('body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        features = []

        if head_neg[0] >= 0:
            features.append(head_neg[1][0])
        else:
            features.append(-1)
        #The following section has been previously commented out - I do not know anymore why this has been done..
        if body_neg[0] >= 0:
            features.append(body_neg[1][0])
        else:
            features.append(-1)

        return features

    x = []
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_negation(headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x


def stanford_negation_features_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_negation_features(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_negation_features_2sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=2):
    return stanford_negation_features(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_negation_features_3sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=3):
    return stanford_negation_features(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_avg_words_per_sent(headlines, bodies, bodyIds, headIds, order_sentences=False, num_sents=99):
    myStanfordmethods = StanfordMethods()
    mytf_tf_idf_helpers = tf_idf_helpers()

    def calculate_words_per_sent(headline, body, body_id, head_id):
        clean_headline = clear_unwanted_chars(headline)
        clean_body = clear_unwanted_chars(body)

        ranked_sentences, body_id = stanford_helper_order_sents(order_sentences, num_sents, body_id,clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers)

        headline_nouns, headline_verbs, head_neg, head_sentiment, head_words_per_sentence = myStanfordmethods.getStanfordInfo('headline', str(body_id), str(head_id), clean_headline, max_number_of_sentences=num_sents)
        body_nouns, body_verbs, body_neg, body_sentiment, body_words_per_sentence = myStanfordmethods.getStanfordInfo('body', str(body_id), str(head_id), ranked_sentences, max_number_of_sentences=num_sents)

        features = []
        features.append(head_words_per_sentence)
        features.append(body_words_per_sentence)

        return features

    x = []
    for i, (headline, body, bodyIds, headIds) in tqdm(enumerate(zip(headlines, bodies, bodyIds, headIds))):
        x.append(calculate_words_per_sent(headline, body, bodyIds, headIds))
    # save all information in file
    myStanfordmethods.store_pickle_file()
    return x


def stanford_avg_words_per_sent_1sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=1):
    return stanford_avg_words_per_sent(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_avg_words_per_sent_2sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=2):
    return stanford_avg_words_per_sent(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

def stanford_avg_words_per_sent_3sent(headlines, bodies, bodyIds, headIds, order_sentences=True, num_sents=3):
    return stanford_avg_words_per_sent(headlines, bodies, bodyIds, headIds, order_sentences, num_sents)

'This is not a feature, but used by any stanford feature calculation to order the sentences based on their tf idf score'
def stanford_helper_order_sents(order_sentences, num_of_sents, body_id, clean_headline, clean_body, myStanfordmethods, mytf_tf_idf_helpers):
    #Order sentences by tf-idf-score:
    if order_sentences:
        body_id = "ranked_"+str(num_of_sents)+"_"+str(body_id)
        'Only rank sentences, if there is no entry in the StanfordPickle'
        if not myStanfordmethods.check_if_already_parsed(body_id):
            #print(body_id + " is not in stanford_pickle")
            ranked_sentences = mytf_tf_idf_helpers.order_by_tf_id_rank(clean_headline, clean_body, num_of_sents)
        else:
            'In this case the content of ranked sentences does not matter, since the Stanford stored information is used'
            #print(body_id + " is already in stanford_pickle _ skipping tf_idf_ranking")
            ranked_sentences = clean_body
    else:
        ranked_sentences = clean_body
        body_id = "unranked_"+str(body_id)

    return ranked_sentences, body_id


def discuss_features(headlines, bodies):
    _discuss_words = [
        'allegedly',
        'report',
        'reported',
        'reportedly',
        'said',
        'say',
        'source',
        'sources',
        'told',
        'according to',
        'claim',
        'claims'
    ]

    def calculate_discuss_feature(text):
        tokens = get_tokenized_lemmas(text)
        #result = [1 if word in tokens else 0 for word in _discuss_words]
        result = min(1,sum([t in _discuss_words for t in tokens]))
        return result

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_discuss_feature(clean_headline))
        features.append(calculate_discuss_feature(clean_body))
        X.append(features)
    print(str(len(X)))
    return np.array(X)


## Benjamins LSTM features:
def single_flat_LSTM_50d_100(headlines, bodies, fold):
    # Following the guide at https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # see also documentation https://keras.io/layers/embeddings/

    """
    Improve on former LSTM features by dividing the tokens much better on the documents and evidences for a claim, in order to remove sparsitiy
    and add more useful information into the vectors.
    :param claims:
    :param evidences:
    :param orig_docs:
    :param fold:
    :return:
    """
    from fnc.refs.feature_engineering_helper.misc import create_embedding_lookup_pandas, \
        text_to_sequences_fixed_size, load_embedding_pandas

    #########################
    # PARAMETER DEFINITIONS #
    #########################
    method_name = "single_flat_LSTM_50d_100"
    # location path for features
    FEATURES_DIR = "%s/../data/fnc-1/features/" % (path.dirname(path.dirname(path.abspath(__file__))))
    PARAM_DICT_FILENAME = method_name+"_param_dict.pkl"

    param_dict = {
        "MAX_NB_WORDS": 50000,  # size of the vocabulary

        # sequence lengths
        "MAX_SEQ_LENGTH": 100, #1000

        # embedding specific values
        "EMBEDDING_DIM": 50,  # dimension of the GloVe embeddings
        "GLOVE_ZIP_FILE": 'glove.twitter.27B.zip',
        "GLOVE_FILE": 'glove.twitter.27B.50d.txt',

        # embedding file names
        "EMBEDDING_FILE": method_name+"_embedding.npy",

        # vocab file names
        "VOCAB_FILE": method_name+"_vocab.pkl",
    }


    ###############################################
    # GET VOCABULARY AND PREPARE EMBEDDING MATRIX #
    ###############################################

    # load GloVe embeddings
    GloVe_vectors = load_embedding_pandas(param_dict["GLOVE_ZIP_FILE"], param_dict["GLOVE_FILE"])

    # load all claims, orig_docs and evidences
    all_heads, all_bodies = word_ngrams.get_head_body_tuples(include_holdout=True)
    all = all_heads
    all.extend(all_bodies)


    # Comment out for clean ablation checks
    # add the unlabeled test data words to the BoW of test+train+holdout data
    h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_unlbled_test()
    all.extend(h_unlbled_test)
    all.extend(b_unlbled_test)

    # create and save the embedding matrices for claims, orig_docs and evidences
    vocab = create_embedding_lookup_pandas(all, param_dict["MAX_NB_WORDS"], param_dict["EMBEDDING_DIM"],
                                           GloVe_vectors, param_dict["EMBEDDING_FILE"], param_dict["VOCAB_FILE"], init_zeros=False,
                                           add_unknown=True, rdm_emb_init=True, tokenizer=nltk.word_tokenize)

    # unload GloVe_vectors in order to make debugging possible
    del GloVe_vectors


    #################################################
    # Create sequences and embedding for the claims #
    #################################################
    print("Create sequences and embedding for the heads")

    concatenated = []
    for i in range(len(headlines)):
        concatenated.append(headlines[i] + ". " + bodies[i])

    # replace tokens of claims by vocabulary ids - the ids refer to the index of the embedding matrix which holds the word embedding for this vocab word
    sequences = text_to_sequences_fixed_size(concatenated, vocab, param_dict["MAX_SEQ_LENGTH"], save_full_text=False,
                                             take_full_claim=True)



    #################################################
    # SAVE PARAM_DICT AND CONCATENATE TRAINING DATA #
    #################################################

    # save param_dict
    with open(FEATURES_DIR+PARAM_DICT_FILENAME, 'wb') as f:
        pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)
    print("Save PARAM_DICT as " + FEATURES_DIR+PARAM_DICT_FILENAME)

    return sequences

#### Features from Benjamin

## Helper functions
def get_head_body_tuples_test():
    d = myConstants.testdataset

    h = []
    b = []
    for stance in d.stances:
        h.append(stance['Headline'])
        b.append(d.articles[int(stance['Body ID'])])

    return h, b

def get_head_body_tuples(include_holdout=False):
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    dataset = DataSet(data_path)
    '''
    data_path = myConstants.data_path
    splits_dir = myConstants.splits_dir
    dataset = myConstants.d

    def get_stances(dataset, folds, holdout):
        # Creates the list with a dict {'headline': ..., 'body': ..., 'stance': ...} for each
        # stance in the data set (except for holdout)
        stances = []
        for stance in dataset.stances:
            if stance['Body ID'] in holdout and include_holdout == True:
                stances.append(stance)
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances.append(stance)

        return stances

    # create new vocabulary
    folds, holdout = kfold_split(dataset, n_folds=10, base_dir=splits_dir)  # [[133,1334,65645,], [32323,...]] => body ids for each fold
    stances = get_stances(dataset, folds, holdout)

    print("Stances length: " + str(len(stances)))

    h = []
    b = []
    # create the final lists with all the headlines and bodies of the set except for holdout
    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    return h, b

def get_unigram_features_of_lexicon(headlines, bodies, lexicon_path, no_hash=False, given_lexicon=False):
    def polarity(x):
        score = wordDict[x]
        if score > 0:
            return 'positive'
        if score < 0:
            return 'negative'
        else:
            return 'none'

    def count_tokens_with_polarity(tokenized):

        scorelist = []
        for token in tokenized:
            token = token.lower()
            score = polarity(token)
            scorelist.append(score)

        pol_dict = dict(Counter(scorelist))

        if 'none' not in pol_dict:
            pol_dict['none'] = 0

        if 'positive' not in pol_dict:
            pol_dict['positive'] = 0

        if 'negative' not in pol_dict:
            pol_dict['negative'] = 0

        return pol_dict

    def polarity_sum(tokenized):

        negList = []
        posList = []
        for token in tokenized:
            token = token.lower()
            if polarity(token) == 'positive':
                posList.append(wordDict[token])
            elif polarity(token) == 'negative':
                negList.append(abs(wordDict[token]))

        return {'pos_sum': sum(posList), 'neg_sum': sum(negList)}

    def max_token(tokenized):

        negList = []
        posList = []

        for token in tokenized:
            token = token.lower()
            if polarity(token) == 'positive':
                posList.append(wordDict[token])
            elif polarity(token) == 'negative':
                negList.append(wordDict[token])

        try:
            pos_max = max(posList)
        except ValueError:
            pos_max = 0
        try:
            neg_max = min(negList)
        except ValueError:
            neg_max = 0

        return {'pos_max': pos_max, 'neg_max': neg_max}

    def last_token(tokenized):

        pol_dict = {'last_polarity': 0}

        for token in reversed(tokenized):
            token = token.lower()
            if polarity(token) == 'positive' or polarity(token) == 'negative':
                pol_dict['last_polarity'] = wordDict[token]
            else:
                continue
        return pol_dict

    def all_feats_dict(string, tokenizer):

        tokenized = tokenizer.word_tokenize(string)

        ct = count_tokens_with_polarity(tokenized)
        pol = polarity_sum(tokenized)
        max_tkn = max_token(tokenized)
        last = last_token(tokenized)

        complete = dict()
        for dictionary in [ct, pol, max_tkn, last]:
            complete.update(dictionary)
        return complete

    wordDict = defaultdict(float)
    if given_lexicon == False:
        # load lexicon and create dictionary out of it
        with open(lexicon_path, 'r') as f:
            for row in f.readlines():
                row = row.split()
                if (no_hash and row[0].startswith('#')):
                    row[0] = row[0][1:]
                wordDict[row[0]] = float(row[1])
    else:
        wordDict = given_lexicon

    # create features and build feature matrix
    emo_counts_head = [all_feats_dict(headline, nltk) for headline in tqdm(headlines)]
    emo_counts_body = [all_feats_dict(body, nltk) for body in tqdm(bodies)]

    emo_counts_head_df = pd.DataFrame(emo_counts_head)
    emo_counts_head_df = emo_counts_head_df.fillna(0)

    emo_counts_body_df = pd.DataFrame(emo_counts_body)
    emo_counts_body_df = emo_counts_body_df.fillna(0)

    emo_counts = np.concatenate([emo_counts_head_df.as_matrix(), emo_counts_body_df.as_matrix()], axis=1)

    return emo_counts

def char_3grams_5000_concat_all_data(headlines, bodies):

    def combine_head_and_body(headlines, bodies):
        return [headline + " " + body for i, (headline, body) in
                tqdm(enumerate(zip(headlines, bodies)))]

    # Load train data into CountVectorizer, get the resulting X-values and also the vocabulary
    # for the test data feature creation
    def get_features(headlines, bodies, headlines_all, bodies_all):
        # create vocab on basis of training data
        head_and_body = combine_head_and_body(headlines_all, bodies_all)
        head_and_body_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=True,
                                              max_features=5000, use_idf=False, norm='l2')
        head_and_body_tfidf.fit(head_and_body)
        vocab = head_and_body_tfidf.vocabulary_

        # create training feature vectors
        X_train_head_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=True,
                                             stop_words='english', vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = X_train_head_tfidf.fit_transform(headlines)

        X_train_body_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=True,
                                             stop_words='english', vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = X_train_body_tfidf.fit_transform(bodies)

        X_train = np.concatenate([X_train_head.toarray(), X_train_body.toarray()], axis=1)

        return X_train

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()

    # Comment out for clean ablation tests
    h.extend(h_test)
    b.extend(b_test)

    X_train = get_features(headlines, bodies, h, b)

    return X_train

def lexical_features(headlines, bodies):
    # calculates lexical diversities for head and body

    def get_info(text):
        sent_list = nltk.sent_tokenize(text)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sent_list]

        word_count = 0
        punctuation_count = 0
        types = set()
        token_list = []
        for sent in sent_list:
            for word in sent:
                token_list.extend(word)
                # get number of punctuations
                if word in string.punctuation:
                    punctuation_count += 1
                else:
                    # get types
                    types.add(word.lower())

                    # get number of tokens
                    word_count += 1
        return types, word_count, token_list

    def get_head_features(head):
        types, word_count, _ = get_info(head)

        # get type-toke-ratio TTR (STTR might be better http://www.lexically.net/downloads/version5/HTML/index.html?type_token_ratio_proc.htm)
        if word_count != 0:
            ttr = float(len(types) / word_count)
        else:
            ttr = 0

        return [ttr]

    def get_body_features(body):
        types, word_count, token_list = get_info(body)


        # get type-toke-ratio TTR (STTR might be better http://www.lexically.net/downloads/version5/HTML/index.html?type_token_ratio_proc.htm)
        if word_count != 0:
            ttr = float(len(types) / word_count)
        else:
            ttr = 0

        # lexical diversity
        mtld = fe_util.mtld(token_list)

        return [ttr, mtld]

    body_features = [get_body_features(body) for body in tqdm(bodies)]
    head_features = [get_head_features(head) for head in tqdm(headlines)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features

def max_diff_twitter_uni_bigrams(headlines, bodies):
    # Generates a set of features from the MaxDiff Twitter Sentiment Lexicon.
    # Features generated follow the ones generated in
    # [Mohammad et. al 2013](http://www.aclweb.org/website/old_anthology/S/S13/S13-2.pdf#page=357)
    # - The polarity occurences (neg, none, pos) of all tokens and bigrams of the text
    # - Sum of score within tweet for each `p`
    # - Maximum token score for each `p`
    # - Score of last token in each tweet
    #
    # Source:  http://saifmohammad.com/WebPages/lexicons.html#EmoLex4


    def last_token(tokenized, ngrams_list):
        # retrieve the polarity of the last unigram or bigram and take the highest value

        for unigram, bigram in list(zip_longest(reversed(tokenized), reversed(ngrams_list))):
            if unigram is not None:
                unigram = unigram.lower()

            if bigram is not None:
                bigram = bigram.lower()

            if polarity(unigram) != 'none' or polarity(bigram) != 'none':
                try:
                    last_polarity_uni = wordDict[unigram]
                except KeyError:
                    last_polarity_uni = 0
                try:
                    last_polarity_bi = wordDict[bigram]
                except KeyError:
                    last_polarity_bi = 0

                if abs(last_polarity_uni) > abs(last_polarity_bi):
                    return {'last_polarity': last_polarity_uni}
                elif abs(last_polarity_uni) < abs(last_polarity_bi):
                    return {'last_polarity': last_polarity_bi}
                elif abs(last_polarity_uni) == abs(last_polarity_bi):
                    return {'last_polarity': last_polarity_uni}
                else:
                    return {'last_polarity': 0}
            else:
                continue

        else:  # called if KeyError occures
            return {'last_polarity': 0}

    def max_token(tokenized):
        # get highest and lowest polarity value for the words in the text
        negList = []
        posList = []

        for token in tokenized:
            token = token.lower()
            if polarity(token) == 'positive':
                posList.append(wordDict[token])
            elif polarity(token) == 'negative':
                negList.append(wordDict[token])

        try:
            pos_max = max(posList)
        except ValueError:
            pos_max = 0
        try:
            neg_max = min(negList)
        except ValueError:
            neg_max = 0

        return {'pos_max': pos_max, 'neg_max': neg_max}

    def polarity_sum(all_grams):
        # sums up the polarity-values found in the lexicon for the
        # tokens and bigrams in a text (negative and positive separately)
        negList = []
        posList = []

        for token in all_grams:
            token = token.lower()
            if polarity(token) == 'positive':
                posList.append(wordDict[token])
            elif polarity(token) == 'negative':
                negList.append(abs(wordDict[token]))

        return {'pos_sum': sum(posList), 'neg_sum': sum(negList)}

    def polarity(x):
        score = wordDict[x]
        if score > 0:
            return 'positive'
        if score < 0:
            return 'negative'
        else:
            return 'none'

    def count_tokens_with_polarity(all_grams):
        # counts the polarity (neg, none, pos) of all tokens and bigrams of the text
        scorelist = []
        for token in all_grams:
            token = token.lower()
            score = polarity(token)
            scorelist.append(score)

        pol_dict = dict(Counter(scorelist))

        if 'none' not in pol_dict:
            pol_dict['none'] = 0

        if 'positive' not in pol_dict:
            pol_dict['positive'] = 0

        if 'negative' not in pol_dict:
            pol_dict['negative'] = 0

        return pol_dict

    def get_function_parameters(string, tokenizer):
        tokenized = tokenizer.word_tokenize(string)
        ngrams_list = [' '.join(i) for i in nltk.ngrams(tokenized, 2)]
        all_grams = tokenized + ngrams_list

        return tokenized, ngrams_list, all_grams

    def all_feats_dict(string, tokenizer):

        tokenized, ngrams_list, all_grams = get_function_parameters(string, tokenizer)

        ct = count_tokens_with_polarity(all_grams)
        pol = polarity_sum(all_grams)
        max_tkn = max_token(tokenized)
        last = last_token(tokenized, ngrams_list)

        complete = dict()

        for dictionary in [ct, pol, max_tkn, last]:
            complete.update(dictionary)
        return complete

    lexicon_path = "%s/../data/lexicons/maxDiffTwitter/" % (path.dirname(path.dirname(path.abspath(__file__))))
    wordDict = defaultdict(float)
    with open(lexicon_path + 'SemEval2015-English-Twitter-Lexicon.txt', 'r') as f:
        for row in f.readlines():
            row = row.split()
            wordDict[' '.join(row[1:])] = float(row[0])

    emo_counts_head = [all_feats_dict(headline, nltk) for headline in tqdm(headlines)]
    emo_counts_body = [all_feats_dict(body, nltk) for body in tqdm(bodies)]

    emo_counts_head_df = pd.DataFrame(emo_counts_head)
    emo_counts_head_df = emo_counts_head_df.fillna(0)

    emo_counts_body_df = pd.DataFrame(emo_counts_body)
    emo_counts_body_df = emo_counts_body_df.fillna(0)

    emo_counts = np.concatenate([emo_counts_head_df.as_matrix(), emo_counts_body_df.as_matrix()], axis=1)

    return emo_counts

def mpqa_unigrams(headlines, bodies):
    """
    Extracts the same features as in get_unigram_features_of_lexicon, just with the subjectivity clues lexicon

    NOTE:
        Simplified form of dictionary initialization here; an entry for a word can be an array. E.g.
        dict["abandon"] = [(-1, strongsubj), (-1, weaksubj)]. This is ignored here. The strongsubj entries
        are getting multiplied by a factor 3 and will replace weaksubj entries.

    ADDITIONAL NOTE:
        Weighted features perform worse (See feature_engineering_crapyard.py variants of this method)

    """

    lexicon_path = "%s/../data/lexicons/MPQA/subjectivity-clues.csv" % (
        path.dirname(path.dirname(path.abspath(__file__))))

    wordDict = defaultdict(float)
    with open(lexicon_path, 'r') as f:
        reader = csv.reader(f)
        headerRows = [i for i in range(0, 1)]
        for row in headerRows:
            next(reader)
        for row in reader:

            score = row[5]
            if score == 'positive':
                score = 1
            elif score == 'negative':
                score = -1
            else:
                score = 0

            subjectivity = row[0]
            if subjectivity == 'strongsubj':
                score = score * 3

            if row[2] in wordDict and wordDict[row[2]] < score:
                wordDict[row[2]] = float(score)
            elif row[2] not in wordDict:
                wordDict[row[2]] = float(score)

    return get_unigram_features_of_lexicon(headlines, bodies, "", no_hash=False, given_lexicon=wordDict)

def negated_context_word_12grams_concat_tf5000_l2_all_data(headlines, bodies):
    """
    Negates string after special negation word by adding a "NEG_" in front
    of every negated word, until a punctuation mark appears.
    Source:
        NRC-Canada: Buidling the State-of-the-Art in Sentiment Analysis of Tweets
        http://sentiment.christopherpotts.net/lingstruc.html
        http://stackoverflow.com/questions/23384351/how-to-add-tags-to-negated-words-in-strings-that-follow-not-no-and-never


    :param headlines:
    :param bodies:
    :return:
    """

    def get_negated_text(text):
        transformed = re.sub(
            r'\b'
            r'(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|'
            r'cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|'
            r'aint|[a-z].n\'t)'
            r'\b'
            r'[\w\s]+[,.:;!?]',
            lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)),
            text,
            flags=re.IGNORECASE)
        return transformed

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_vocab(neg_headlines, neg_bodies):
        tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000, use_idf=False,
                                        norm='l2')
        tf_vectorizer.fit_transform(combine_head_and_body(neg_headlines, neg_bodies))
        vocab = tf_vectorizer.vocabulary_

        return vocab

    def get_features(neg_headlines_test, neg_bodies_test, vocab):
        tf_vectorizer_head = TfidfVectorizer(vocabulary=vocab, stop_words='english', use_idf=False, norm='l2')
        X_test_head = tf_vectorizer_head.fit_transform(neg_headlines_test)

        tf_vectorizer_body = TfidfVectorizer(vocabulary=vocab, stop_words='english', use_idf=False, norm='l2')
        X_test_body = tf_vectorizer_body.fit_transform(neg_bodies_test)

        X_test = np.concatenate([X_test_head.toarray(), X_test_body.toarray()], axis=1)
        return X_test

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()

    # Comment out for clean ablation tests
    h.extend(h_test)
    b.extend(b_test)

    neg_headlines_all = [get_negated_text(h) for h in h]
    neg_bodies_all = [get_negated_text(b) for b in b]
    neg_headlines = [get_negated_text(h) for h in headlines]
    neg_bodies = [get_negated_text(b) for b in bodies]

    vocab = get_vocab(neg_headlines_all, neg_bodies_all)
    X_train = get_features(neg_headlines, neg_bodies, vocab)

    return X_train

def nrc_emo_lex(headlines, bodies):
    """
    Counts Number of words in a text associated with 8 different emotions.
    Uses EmoLex lexicon: http://saifmohammad.com/WebPages/lexicons.html#EmoLex

    """

    lexicon_path = "%s/../data/lexicons/emoLex/" % (path.dirname(path.dirname(path.abspath(__file__))))
    word_list = defaultdict(list)
    # emotion_list = defaultdict(list)
    emotion_set = set()

    with open(lexicon_path + 'NRC_emotion_lexicon_list.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for word, emotion, present in reader:
            if int(present) == 1:  # 1 = word/emotion-allocation present
                word_list[word].append(emotion)
                # emotion_list[emotion].append(word)
                emotion_set.add(emotion)

    def generate_emotion_count(string):
        emo_count = Counter()
        for token in nltk.word_tokenize(string):
            token = token.lower()
            emo_count += Counter(word_list[token])

        # Guarantee same length for each feature vector by adding emotions
        # that do no appear in the text
        for emotion in emotion_set:
            if (emotion not in emo_count):
                emo_count[emotion] = 0

        return emo_count

    emo_counts_head = [generate_emotion_count(headline) for headline in tqdm(headlines)]
    emo_counts_body = [generate_emotion_count(body) for body in tqdm(bodies)]

    emo_counts_head_df = pd.DataFrame(emo_counts_head)
    emo_counts_head_df = emo_counts_head_df.fillna(0)

    emo_counts_body_df = pd.DataFrame(emo_counts_body)
    emo_counts_body_df = emo_counts_body_df.fillna(0)

    emo_counts = np.concatenate([emo_counts_head_df.as_matrix(), emo_counts_body_df.as_matrix()], axis=1)

    return emo_counts


def nrc_hashtag_sentiment_unigram(headlines, bodies):
    lexicon_path = "%s/../data/lexicons/hashtagSentiment/unigrams-pmilexicon.txt" % (
        path.dirname(path.dirname(path.abspath(__file__))))
    return get_unigram_features_of_lexicon(headlines, bodies, lexicon_path, no_hash=False)


def nrc_hashtag_sentiment_unigram_POS(headlines, bodies):
    """
    From Paper: NRC-Canada: Building the State-Of-The-Art in Sentiment Analysis of Tweets
                Mohammad et al.
    # calculates the nrc hashtag sentiments for:
    # all verbs in the head / body
    # all nouns in the head / body
    # all adjectives in head / body
    # and merges the features after that

    """

    def get_features_head(headline):
        tokenized_head = nltk.word_tokenize(headline)
        pos_tags_head = nltk.pos_tag(tokenized_head, tagset='universal')

        head_text_VERB = ""
        head_text_ADJ = ""
        head_text_NOUN = ""
        head_text_ADV = ""
        head_text_PRON = ""
        for word, tag in pos_tags_head:
            if tag == 'VERB':
                head_text_VERB += " " + word
            if tag == 'ADJ':
                head_text_ADJ += " " + word
            if tag == 'NOUN':
                head_text_NOUN += " " + word
            if tag == 'ADV':
                head_text_ADV += " " + word
            if tag == 'PRON':
                head_text_PRON += " " + word
        pos_dict_head['VERB'].append(head_text_VERB)
        pos_dict_head['ADJ'].append(head_text_ADJ)
        pos_dict_head['NOUN'].append(head_text_NOUN)
        pos_dict_head['ADV'].append(head_text_ADV)
        pos_dict_head['PRON'].append(head_text_PRON)

    def get_features_body(body):
        sent_list_body = nltk.sent_tokenize(body)
        tokenized_sents_body = [nltk.word_tokenize(sent) for sent in sent_list_body]
        pos_tags = nltk.pos_tag_sents(tokenized_sents_body, tagset='universal')

        text_VERB = ""
        text_ADJ = ""
        text_NOUN = ""
        text_ADV = ""
        text_PRON = ""

        for sent in pos_tags:
            for word, tag in sent:
                if tag == 'VERB':
                    text_VERB += " " + word
                if tag == 'ADJ':
                    text_ADJ += " " + word
                if tag == 'NOUN':
                    text_NOUN += " " + word
                if tag == 'ADV':
                    text_ADV += " " + word
                if tag == 'PRON':
                    text_PRON += " " + word
        pos_dict_body['VERB'].append(text_VERB)
        pos_dict_body['ADJ'].append(text_ADJ)
        pos_dict_body['NOUN'].append(text_NOUN)
        pos_dict_body['ADV'].append(text_ADV)
        pos_dict_body['PRON'].append(text_PRON)

    pos_dict_head = {
        'VERB': [],
        'ADJ': [],
        'NOUN': [],
        'ADV': [],
        'PRON': []
    }
    pos_dict_body = {
        'VERB': [],
        'ADJ': [],
        'NOUN': [],
        'ADV': [],
        'PRON': []
    }

    for h in tqdm(headlines):
        get_features_head(h)

    for b in tqdm(bodies):
        get_features_body(b)

    verb_features = nrc_hashtag_sentiment_unigram(pos_dict_head['VERB'], pos_dict_body['VERB'])
    adj_features = nrc_hashtag_sentiment_unigram(pos_dict_head['ADJ'], pos_dict_body['ADJ'])
    noun_features = nrc_hashtag_sentiment_unigram(pos_dict_head['NOUN'], pos_dict_body['NOUN'])
    adv_features = nrc_hashtag_sentiment_unigram(pos_dict_head['ADV'], pos_dict_body['ADV'])
    pron_features = nrc_hashtag_sentiment_unigram(pos_dict_head['PRON'], pos_dict_body['PRON'])

    feature_matrix = np.concatenate(
        [verb_features,
         adj_features,
         noun_features,
         adv_features,
         pron_features
         ], axis=1)

    return feature_matrix

def sentiment140_unigrams(headlines, bodies):
    lexicon_path = "%s/../data/lexicons/sentiment140/unigrams-pmilexicon.txt" % (
        path.dirname(path.dirname(path.abspath(__file__))))
    return get_unigram_features_of_lexicon(headlines, bodies, lexicon_path, no_hash=False)


def readability_features(headlines, bodies):

    def get_head_features(head):
        tokenized = nltk.word_tokenize(head)
        word_counter = 0

        # get average word length
        for word in tokenized:
            if word not in string.punctuation:
                word_counter += 1

        # get coleman liau index
        CL_index = fe_util.coleman_liau_index(head, word_counter)

        # get automated readability index
        AR_index = fe_util.automated_readability_index(head, word_counter)

        # LIX readability index
        LIX_index = fe_util.lix_index(head, word_counter)

        # RIX readability index
        RIX_index = fe_util.rix_index(head)

        # McAlpine EFLAW index
        EFLAW_index = fe_util.mcalpine_eflaw_index(head)

        # Strain index
        strain_index = fe_util.strain_index(head)

        # calculate flesch-kincaid grade level
        FK_grade = fe_util.flesch_grade_level(head)

        # calculate gunning-fog grade level
        GI_grade = fe_util.gunning_fog_index(head, word_counter)

        # Flesh Kincaid Reading Ease index
        FK_reading_ease = fe_util.flesch_reading_ease(head)

        return [FK_grade, GI_grade, FK_reading_ease, CL_index, AR_index, LIX_index, RIX_index, EFLAW_index, strain_index]

    def get_body_features(body):
        # get number of nouns and tokens
        sent_list = nltk.sent_tokenize(body)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sent_list]
        pos_tags_sents = nltk.pos_tag_sents(tokenized_sents)

        word_count = 0
        punctuation_count = 0
        types = set()
        for sent in pos_tags_sents:
            for word, tag in sent:
                # get number of punctuations
                if word in string.punctuation:
                    punctuation_count += 1
                else:
                    # get types
                    types.add(word.lower())

                    # get number of tokens
                    word_count += 1


        # get coleman liau index
        CL_index = fe_util.coleman_liau_index(body, word_count)

        # get automated readability index
        AR_index = fe_util.automated_readability_index(body, word_count)

        # LIX readability index
        LIX_index = fe_util.lix_index(body, word_count)

        # RIX readability index
        RIX_index = fe_util.rix_index(body)

        # McAlpine EFLAW index
        EFLAW_index = fe_util.mcalpine_eflaw_index(body)

        # Strain index
        strain_index = fe_util.strain_index(body)

        # calculate flesch-kincaid grade level
        FK_grade = fe_util.flesch_grade_level(body)

        # calculate gunning-fog grade level
        GI_grade = fe_util.gunning_fog_index(body, word_count)

        SMOG_index = fe_util.smog_index(body)

        # Flesh Kincaid Reading Ease index
        FK_reading_ease = fe_util.flesch_reading_ease(body)

        return [FK_grade, GI_grade, FK_reading_ease, CL_index, AR_index, LIX_index, RIX_index, EFLAW_index, strain_index, SMOG_index]

    body_features = [get_body_features(body) for body in tqdm(bodies)]
    head_features = [get_head_features(head) for head in tqdm(headlines)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features

def structural_features(headlines, bodies):
    """
    Implements the significant features of the paper "This Just In: Fake News Packs a Lot in Title,
    Uses Simpler, Repetitive Content in Text Body, More Similar to Satire than Real News" by
    Benjamin D. Horne and Sibel Adali of Rensslar Polytechnic Institute, New York

    Open to implement:
        avg_negstr: method implemented, but due to opening a jar lib way too slow to use
        analytic: LIWC lexicon needed (not free)
    """

    def count_verb_phrases(t, print_tree=False):
        # http: // www.nltk.org / book / ch07.html  # developing-and-evaluating-chunkers
        count = 0
        try:
            t.label()
        except AttributeError:
            if print_tree:
                print(t, end=" ")
        else:
            # Now we know that t.node is defined
            if print_tree:
                print('(', t.label(), end=" ")
            for child in t:
                if t.label() == 'VP':
                    count = 1
                count = count + count_verb_phrases(child, print_tree)
            if print_tree:
                print(')', end=" ")
        return count

    def get_head_features(head):
        tokenized = nltk.word_tokenize(head)
        word_len_sum = 0
        avg_wlen = 0
        word_counter = 0

        # get average word length
        for word in tokenized:
            if word not in string.punctuation:
                word_len_sum += len(word)
                word_counter += 1
        if word_counter > 0:
            avg_wlen = float(word_len_sum / word_counter)
        return [avg_wlen]

    def get_paragraph_breaks(text):
        """Identifies indented text or line breaks as the beginning of
        paragraphs and returns a list with indices of paragraph
        beginnings. List always starts with a 0 => from TextTilingTokenizer"""

        MIN_PARAGRAPH = 100
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start() - last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks

    def get_avg_paragraph_length(text, pbreaks):
        """
        Takes a text and the indices of the paragraph breaks and reaturn the average
        paragraph lengths
        """
        paragraph_list = []
        counter = 0
        for index in pbreaks:
            if counter > 0:
                paragraph_list.append(text[pbreaks[counter - 1]:index])
            counter += 1
        paragraph_list.append(text[pbreaks[-1]:])

        paragraph_lengths = []
        for para in paragraph_list:
            tokenized = nltk.word_tokenize(para)
            para_length = 0
            for token in tokenized:
                if token not in string.punctuation:
                    para_length += 1
            paragraph_lengths.append(para_length)

        if len(paragraph_lengths) > 0:
            return sum(paragraph_lengths) / len(paragraph_lengths)
        else:
            return 0

    def get_body_features(body):

        # get number of nouns and tokens
        sent_list = nltk.sent_tokenize(body)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sent_list]


        word_count = 0
        punctuation_count = 0

        word_len_sum = 0
        for sent in tokenized_sents:
            for word in sent:

                # get number of punctuations
                if word in string.punctuation:
                    punctuation_count += 1
                else:
                    # sum up length of words
                    word_len_sum += len(word)

                    # get number of tokens
                    word_count += 1


        # number of paragraphs and their avg lengths
        pbreaks = get_paragraph_breaks(body)
        paragraph_count = len(pbreaks) - 1
        avg_paragraph_length = get_avg_paragraph_length(body, pbreaks)

        # get average word length
        avg_wlen = 0
        if word_count > 0:
            avg_wlen = float(word_len_sum / word_count)

        return [avg_wlen, paragraph_count, avg_paragraph_length]

    body_features = [get_body_features(body) for body in tqdm(bodies)]
    head_features = [get_head_features(head) for head in tqdm(headlines)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features

##\\Too add explanation
def POS_features(headlines, bodies):
    """
    Implements the significant features of the paper "This Just In: Fake News Packs a Lot in Title,
    Uses Simpler, Repetitive Content in Text Body, More Similar to Satire than Real News" by
    Benjamin D. Horne and Sibel Adali of Rensslar Polytechnic Institute, New York

    Open to implement:
        avg_negstr: method implemented, but due to opening a jar lib way too slow to use
        analytic: LIWC lexicon needed (not free)
    """

    def count_verb_phrases(t, print_tree=False):
        # http: // www.nltk.org / book / ch07.html  # developing-and-evaluating-chunkers
        count = 0
        try:
            t.label()
        except AttributeError:
            if print_tree:
                print(t, end=" ")
        else:
            # Now we know that t.node is defined
            if print_tree:
                print('(', t.label(), end=" ")
            for child in t:
                if t.label() == 'VP':
                    count = 1
                count = count + count_verb_phrases(child, print_tree)
            if print_tree:
                print(')', end=" ")
        return count

    def get_head_features(head):
        tokenized = nltk.word_tokenize(head)
        word_len_sum = 0
        avg_wlen = 0
        word_counter = 0

        # get average word length
        for word in tokenized:
            if word not in string.punctuation:
                word_len_sum += len(word)
                word_counter += 1

        if word_counter > 0:
            avg_wlen = float(word_len_sum / word_counter)

        # calculate percentage of stopwords
        stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
        stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
        all_stop_words = stop_words_sklearn.union(stop_words_nltk)
        stop_words_counter = 0
        per_stop = 0
        word_freq_in_head = defaultdict(int)
        for word in tokenized:
            if word.lower() in all_stop_words:
                stop_words_counter += 1
            word_freq_in_head[word] = word_freq_dict[word]
        if word_counter > 0:
            per_stop = stop_words_counter / word_counter

        # calculate frequency of 3 least common words
        w_freq_list = list(reversed(collections.Counter(word_freq_in_head.values()).most_common()))
        flu_reuters_c = 0
        counter = 0
        for i in range(3):
            if len(w_freq_list) > i:
                counter += 1
                flu_reuters_c += w_freq_list[i][0]
        if counter > 0:
            flu_reuters_c = float(flu_reuters_c / counter)
        else:
            flu_reuters_c = 0

        # get number of quotes http://stackoverflow.com/questions/28037857/how-to-extract-all-quotes-in-a-document-text-using-regex
        # and then calculate the ratio #quoted words / #words
        match = re.findall('(?:[\â€œ\'\"](.*?)[\â€\'\"])', head)
        quoted_words_count = 0
        quoted_word_ratio = 0
        for quote in match:
            tokenized_quote = nltk.word_tokenize(quote)
            for token in tokenized_quote:
                if token not in string.punctuation:
                    quoted_words_count += 1
        if word_counter > 0:
            quoted_word_ratio = float(quoted_words_count / word_counter)

        # calculate number of nouns and proper nouns
        pos_tagged = nltk.pos_tag(tokenized)
        NN_count = 0
        NNP_count = 0
        focuspast = 0
        for word, tag in pos_tagged:  # http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
            if tag == 'NN' or tag == 'NNS':
                NN_count += 1
            if tag == 'NNP' or tag == 'NNPS':
                NNP_count += 1
            if tag == 'VBD':  # verb, past tense
                focuspast += 1

        # calculate verb phrases
        grammar = """
            P: {<IN>}           # Preposition
            PP: {<P> <NP>}      # PP -> P NP
            V: {<V.*>}          # Verb
            NP: {<DT>? <JJ>* <NN>*} # NP
            VP: {<RB.*>? <V>+ <NP|PP>*}
        """
        cp = nltk.RegexpParser(grammar)
        sent_list = nltk.sent_tokenize(head)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sent_list]
        pos_tags_sents = nltk.pos_tag_sents(tokenized_sents)

        vp_count = 0
        for pos_sent in pos_tags_sents:
            tree = cp.parse(pos_sent)
            vp_count += count_verb_phrases(tree, print_tree=False)

        return [per_stop, NN_count, NNP_count, vp_count, flu_reuters_c, focuspast, quoted_word_ratio]

    def get_paragraph_breaks(text):
        """Identifies indented text or line breaks as the beginning of
        paragraphs and returns a list with indices of paragraph
        beginnings. List always starts with a 0 => from TextTilingTokenizer"""

        MIN_PARAGRAPH = 100
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start() - last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks

    def get_avg_paragraph_length(text, pbreaks):
        """
        Takes a text and the indices of the paragraph breaks and reaturn the average
        paragraph lengths
        """
        paragraph_list = []
        counter = 0
        for index in pbreaks:
            if counter > 0:
                paragraph_list.append(text[pbreaks[counter - 1]:index])
            counter += 1
        paragraph_list.append(text[pbreaks[-1]:])

        paragraph_lengths = []
        for para in paragraph_list:
            tokenized = nltk.word_tokenize(para)
            para_length = 0
            for token in tokenized:
                if token not in string.punctuation:
                    para_length += 1
            paragraph_lengths.append(para_length)

        if len(paragraph_lengths) > 0:
            return sum(paragraph_lengths) / len(paragraph_lengths)
        else:
            return 0

    def get_body_features(body):
        # get number of quotes http://stackoverflow.com/questions/28037857/how-to-extract-all-quotes-in-a-document-text-using-regex
        match = re.findall('(?:[\â€œ\'\"](.*?)[\â€\'\"])', body)
        quote_count = len(match)

        # get number of nouns and tokens
        sent_list = nltk.sent_tokenize(body)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sent_list]
        pos_tags_sents = nltk.pos_tag_sents(tokenized_sents)

        NN_count = 0
        word_count = 0
        punctuation_count = 0
        PRP_count = 0
        RB_count = 0
        CD_count = 0
        word_len_sum = 0
        types = set()
        for sent in pos_tags_sents:
            for word, tag in sent:

                # get number of punctuations
                if word in string.punctuation:
                    punctuation_count += 1
                else:
                    # sum up length of words
                    word_len_sum += len(word)

                    # get types
                    types.add(word.lower())

                    # get number of tokens
                    word_count += 1

                # get number of personal pronouns
                if tag == 'PRP':
                    PRP_count += 1

                # get number of nouns
                if tag == 'NN' or tag == 'NNS':
                    NN_count += 1

                # get number of nouns
                if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                    RB_count += 1

                if tag == 'CD':
                    CD_count += 1

        # number of paragraphs and their avg lengths
        pbreaks = get_paragraph_breaks(body)
        paragraph_count = len(pbreaks) - 1
        avg_paragraph_length = get_avg_paragraph_length(body, pbreaks)

        # get number of quotes http://stackoverflow.com/questions/28037857/how-to-extract-all-quotes-in-a-document-text-using-regex
        # and then calculate the ratio #quoted words / #words
        match = re.findall('(?:[\â€œ\'\"](.*?)[\â€\'\"])', body)
        quoted_words_count = 0
        quoted_word_ratio = 0
        for quote in match:
            tokenized_quote = nltk.word_tokenize(quote)
            for token in tokenized_quote:
                if token not in string.punctuation:
                    quoted_words_count += 1
        if word_count > 0:
            quoted_word_ratio = float(quoted_words_count / word_count)

        # get average word length
        avg_wlen = 0
        if word_count > 0:
            avg_wlen = float(word_len_sum / word_count)

        return [NN_count,
                punctuation_count, PRP_count, RB_count, CD_count, quoted_word_ratio]

    word_freq_dict = nltk.FreqDist(reuters.words())
    body_features = [get_body_features(body) for body in tqdm(bodies)]
    head_features = [get_head_features(head) for head in tqdm(headlines)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features

####Athene features of the FNC-1
def NMF_fit_all_incl_holdout_and_test(headlines, bodies):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    from sklearn.externals import joblib

    print("WARNING: IF SIZE OF HEAD AND BODY DO NOT MATCH, "
          "RUN THIS FEATURE EXTRACTION METHOD SEPERATELY (WITHOUT ANY OTHER FE METHODS) TO CREATE THE FEATURES ONCE!")

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)


    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_incl_holdout_and_test: fit and transform body")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_incl_holdout_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X

def create_word_ngram_vocabulary(ngram_range=(1,1), max_features=100, lemmatize=False, term_freq=False, norm='l1', use_idf=False, include_holdout=False):
    """
    Creates, returns and saves a vocabulary for (Count-)Vectorizer over all training and test data (holdout excluded) to create BoW
    methods. The method simplifies using the pipeline and later tests with feature creation for a single headline and body.
    This method will cause bleeding, since it also includes the test set.

    :param filename: a filename for the vocabulary
    :param ngram_range: the ngram range for the Vectorizer. Default is (1, 1) => unigrams
    :param max_features: the length of the vocabulary
    :return: the vocabulary
    """
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    dataset = DataSet(data_path)
    '''
    features_dir = myConstants.features_dir

    print("Calling create_word_ngram_vocabulary with ngram_range=("
          + str(ngram_range[0]) + ", " + str(ngram_range[1]) + "), max_features="
          + str(max_features) + ", lemmatize=" +  str(lemmatize) + ", term_freq=" + str(term_freq))
    def get_all_stopwords():
        stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
        stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
        all_stop_words = stop_words_sklearn.union(stop_words_nltk)
        return all_stop_words

    def get_tokenized_lemmas_without_stopwords(s):
        all_stop_words = get_all_stopwords()
        return [normalize_word(t) for t in nltk.word_tokenize(s)
                if t not in string.punctuation and t.lower() not in all_stop_words]


    def train_vocabulary(head_and_body):
        # trains a CountVectorizer on all of the data except for holdout data
        if lemmatize == False:
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features, use_idf=use_idf, norm=norm)
        else:
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features,
                                         tokenizer=get_tokenized_lemmas_without_stopwords)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                             tokenizer=get_tokenized_lemmas_without_stopwords, use_idf=use_idf, norm=norm)
        vectorizer.fit_transform(head_and_body)
        vocab = vectorizer.vocabulary_
        return vocab

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body


    # create filename for vocab
    vocab_file = "word_(" + str(ngram_range[0]) + "_" + str(ngram_range[1]) + ")-gram_" + str(max_features)
    if lemmatize == True:
        vocab_file += "_lemmatized"
    if term_freq == True:
        vocab_file += "_tf"
    if use_idf == True:
        vocab_file += "_idf"
    if include_holdout == True:
        vocab_file += "_holdout"
    vocab_file += "_" + norm + ".pickle"

    # if vocab already exists, just load and return it
    if (os.path.exists(features_dir + "/" + vocab_file)):
        with open(features_dir + "/" + vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)
            print("Existing vocabulary found and load.")
            return vocab

    h, b = get_head_body_tuples(include_holdout=include_holdout)
    head_and_body = combine_head_and_body(h, b) # combine head and body
    vocab = train_vocabulary(head_and_body) # get vocabulary (features)

    # save the vocabulary as file
    with open(features_dir + "/" + vocab_file, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("vocab length: " + str(len(vocab)))
    return vocab

def latent_dirichlet_allocation(headlines, bodies):
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_topics=25, learning_method='online', random_state=0, n_jobs=3)

        print("latent_dirichlet_allocation: fit and transform body")
        t0 = time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time() - t0))

        print("latent_dirichlet_allocation: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        #print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X


    vocab = create_word_ngram_vocabulary(ngram_range=(1, 1), max_features=5000, lemmatize=False, term_freq=True,
                                         norm='l2')
    X = get_features(vocab)
    return X

def latent_dirichlet_allocation_incl_holdout_and_test(headlines, bodies):
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_topics=100, learning_method='online', random_state=0, n_jobs=3)

        print("latent_dirichlet_allocation_incl_holdout_and_test: fit and transform body")
        t0 = time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time() - t0))

        print("latent_dirichlet_allocation_incl_holdout_and_test: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        #print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X


    h, b = get_head_body_tuples(include_holdout=True)

    h_test, b_test = get_head_body_tuples_test()

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000, use_idf=False,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h,b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)
    return X

def latent_semantic_indexing_gensim_holdout_and_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    from gensim import corpora, models

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        all_text = []
        all_text.extend(headlines)
        all_text.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        print("head+body appended size should be around 100k and 19/8k: " + str(len(bodies)))
        head_and_body_tokens = [nltk.word_tokenize(line) for line in all_text]

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

        filename = "lsi_gensim_test_" + str(n_topics) + "topics_and_test"

        h, b = get_head_body_tuples(include_holdout=True)
        h_test, b_test = get_head_body_tuples_test()
        h.extend(h_test)
        b.extend(b_test)
        head_and_body = combine_and_tokenize_head_and_body(h, b,
                                                           file_path=features_dir + "/" + "lsi_gensim_h_b_tokenized_and_test" + ".pkl")

        if (os.path.exists(features_dir + "/" + "lsi_gensim_holdout_and_test" + ".dict")):
            print("dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")
        else:
            print("create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
            corpus_tfidf = tfidf[corpus]

            print("create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)

        return X

    n_topics = 300
    X = get_features(n_topics)

    return X

def NMF_fit_all_concat_300_and_test(headlines, bodies):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            print("X_all_length (w Holdout round 50k): " + str(len(head_and_body)))
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)


    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_concat_300_and_test: fit NMF to all data")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_concat_300_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_concat_300_and_test: concat head and body')
        # calculate cosine distance between the body and head
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X

def latent_semantic_indexing_gensim_test(headlines, bodies):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.

    The differences to the latent_semantic_indexing_gensim are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    from gensim import corpora, models

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        all_text = []
        all_text.extend(headlines)
        all_text.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        print("head+body appended size should be around 100k and 19/8k: " + str(len(bodies)))
        head_and_body_tokens = [nltk.word_tokenize(line) for line in all_text]

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

        filename = "lsi_gensim_test_" + str(n_topics) + "topics"

        h, b = get_head_body_tuples(include_holdout=True)
        head_and_body = combine_and_tokenize_head_and_body(h, b,
                                                           file_path=features_dir + "/" + "lsi_gensim_h_b_tokenized" + ".pkl")

        if (os.path.exists(features_dir + "/" + "lsi_gensim_holdout" + ".dict")):
            print("dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + "lsi_gensim_all" + ".dict")
        else:
            print("create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + "lsi_gensim_all" + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
            corpus_tfidf = tfidf[corpus]

            print("create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)

        return X

    n_topics = 300
    X = get_features(n_topics)

    return X

def NMF_fit_all_concat_300(headlines, bodies):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            print("X_all_length (w Holout round 50k): " + str(len(head_and_body)))
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)


    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all_concat_300"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_concat_300: fit NMF to all data")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_concat_300: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_concat_300: concat head and body')
        # calculate cosine distance between the body and head
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)

    h, b = get_head_body_tuples(include_holdout=True)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X

def word_ngrams_concat_tf5000_l2_w_holdout(headlines, bodies):
    """
    Simple bag of words feature extraction
    """
    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X


    vocab = create_word_ngram_vocabulary(ngram_range=(1,1), max_features=5000,
                                         lemmatize=False, use_idf=False, term_freq=True, norm='l2',
                                         include_holdout=True)

    X = get_features(vocab)

    return X


def word_ngrams_concat_tf5000_l2_w_holdout_and_test(headlines, bodies):
    """
    Simple bag of words feature extraction
    """

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    h, b = get_head_body_tuples(include_holdout=True)
    h_test, b_test = get_head_body_tuples_test()

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h,b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X

def NMF_fit_all(headlines, bodies):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    from sklearn.externals import joblib

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)

    def get_features(head_and_body):
        features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        filename = "NMF_fit_all"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=50, random_state=1, alpha=.1)

            print("NMF_fit_all: fit and transform body")
            t0 = time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X

    h, b = get_head_body_tuples()
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X
