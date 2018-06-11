import zipfile
import numpy as np
import os.path as path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from tqdm import tqdm

FEATURES_DIR = "%s/../data/fnc-1/features/" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
EMBEDDINGS_DIR = "%s/../data/fnc-1/embeddings/" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
DATA_PATH = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
SPLITS_DIR = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))



def load_embedding_pandas(ZIP_FILE, FILE, type="w2v"):
    """
    Loads GloVe embeddings into a dict and returns it
    :param GLOVE_ZIP_FILE: Zip file name of embeddings
    :param GLOVE_FILE: file name inside the zip file
    :return:
    """
    import pandas as pd
    import csv

    # create embedding dict https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    with zipfile.ZipFile(EMBEDDINGS_DIR + ZIP_FILE) as z:
        if type == "w2v":
            embedding = pd.read_table(z.open(FILE), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, skiprows=1)
        else:
            embedding = pd.read_table(z.open(FILE), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        print('Found %s word vectors in GloVe embeddings.' % len(embedding.index))

    return embedding

def create_embedding_lookup_pandas(text_list, max_nb_words, embedding_dim, embedding,
                            embedding_lookup_name, embedding_vocab_name, rdm_emb_init=False, add_unknown=False, tokenizer=None, init_zeros = False):
    """
    Creates the claim embedding lookup table if it not already exists and returns the vocabulary for it
    :param text_list:
    :param max_nb_words:
    :param embedding_dim:
    :param GloVe_vectors:
    :param embedding_lookup_name:
    :param embedding_vocab_name:
    :return:
    """
    #del GloVe_vectors
    if not path.exists(FEATURES_DIR + embedding_lookup_name) or not path.exists(FEATURES_DIR + embedding_vocab_name):
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=None, tokenizer=tokenizer,
                                            max_features=max_nb_words, use_idf=True)
        vectorizer.fit_transform(text_list)
        vocab = vectorizer.vocabulary_


        # do not use 0 since we want to use masking in the LSTM later on
        for word in vocab.keys():
            vocab[word] += 1
        if add_unknown == True:
            max_index = max(vocab.values())
            vocab["UNKNOWN"] = max_index+1

        # prepare embedding - create matrix that holds the glove vector for each vocab entry
        if rdm_emb_init == True:
            embedding_lookup = np.random.random((len(vocab) + 1, embedding_dim))
            zero_vec = np.zeros((embedding_dim))
            embedding_lookup[0] = zero_vec # for masking
        else:
            embedding_lookup = np.zeros((len(vocab) + 1, embedding_dim))

        if init_zeros == False:
            for word, i in vocab.items():
                if word == "UNKNOWN":
                    embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dim)
                    #print(embedding_vector)
                else:
                    try:
                        embedding_vector = embedding.loc[word].as_matrix()
                    except KeyError: #https://stackoverflow.com/questions/15653966/ignore-keyerror-and-continue-program
                        continue
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_lookup[i] = embedding_vector
        #print(embedding_lookup[-1])
        # save embedding matrix
        np.save(FEATURES_DIR + embedding_lookup_name, embedding_lookup)
        # save vocab
        with open(FEATURES_DIR + embedding_vocab_name, 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        print("Embedding lookup table shape for " + embedding_lookup_name + " is: " + str(embedding_lookup.shape))
    else:
        with open(FEATURES_DIR + embedding_vocab_name, "rb") as f:
            vocab = pickle.load(f)

    print("Vocab size for " + embedding_vocab_name + " is: " + str(len(vocab)))

    return vocab

def text_to_sequences_fixed_size(texts, vocab, MAX_SENT_LENGTH, save_full_text=False, take_full_claim = False):
    """
    Turns sentences of claims into sequences of indices provided by the given vocab.
    Unknown words will get an extra index, if
    the vocab has a token "UNKNOWN". The method takes the longest sentence of the claims, if the
    claim should have more than one sentence.
    :param texts:
    :param vocab:
    :param MAX_SENT_LENGTH:
    :return:
    """
    data = np.zeros((len(texts), MAX_SENT_LENGTH), dtype='int32')

    claims = []
    if take_full_claim == False:
        for claim in texts:
            claim_sents = nltk.sent_tokenize(claim)
            word_count_fct = lambda sentence: len(nltk.word_tokenize(sentence)) # take longest sentence of claim if it has more than one
            claims.append(max(claim_sents, key=word_count_fct))
    else:
        claims = texts

    data_string_dict = {}
    for j, claim in tqdm(enumerate(claims)):
        claim_tokens = nltk.word_tokenize(claim.lower())

        data_string = ""
        if save_full_text == True:
            for token in claim_tokens:
                data_string += token + " "
            data_string = data_string[:-1]
            data_string_dict[j] = data_string

        for i, token in enumerate(claim_tokens):
            if i < MAX_SENT_LENGTH:
                index = vocab.get(token, "UNKNOWN")
                if index == "UNKNOWN":
                    index = vocab.get(index, None)
                if index != None:
                    data[j, i] = index

    if save_full_text == True:
        return data, data_string_dict
    else:
        return data
