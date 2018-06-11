import sys, os.path as path
from builtins import isinstance
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from fnc.models.MultiThreadingFeedForwardMLP import MultiThreadingFeedForwardMLP
import datetime
from fnc.utils import printout_manager
from fnc.src.models import Model
import fnc.utils.estimator_definitions as esitmator_definitions
import argparse
import os.path as path
import os
import fnc.utils.score_calculation as score_calculation
import numpy as np
from fnc.refs.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc.refs.feature_engineering import refuting_features, polarity_features, hand_features, word_overlap_features
from fnc.refs.feature_engineering import gen_non_bleeding_feats, gen_or_load_feats
from fnc.refs.feature_engineering import word_unigrams_5000_concat_tf_l2_holdout_unlbled_test, NMF_cos_300_holdout_unlbled_test, \
    NMF_concat_300_holdout_unlbled_test, latent_dirichlet_allocation_25_holdout_unlbled_test, \
    latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test, \
    NMF_cos_50, latent_dirichlet_allocation_25, \
    latent_semantic_indexing_gensim_300_concat_holdout, NMF_concat_300_holdout, word_unigrams_5000_concat_tf_l2_holdout
from fnc.refs.feature_engineering_challenge import NMF_fit_all_incl_holdout_and_test, \
                latent_dirichlet_allocation_incl_holdout_and_test, latent_semantic_indexing_gensim_holdout_and_test,\
                NMF_fit_all_concat_300_and_test, word_ngrams_concat_tf5000_l2_w_holdout_and_test, NMF_fit_all, \
    latent_dirichlet_allocation, latent_semantic_indexing_gensim_test, NMF_fit_all_concat_300, word_ngrams_concat_tf5000_l2_w_holdout
#copied from old implementation
from fnc.refs.feature_engineering import sen2sen_similarity_max, word_mover_distance_similarity_sentence_min, \
    word_mover_distance_wholebody, stanford_ppdb_score, stanford_ppdb_score_1sent, stanford_ppdb_score_2sent, stanford_ppdb_score_3sent, \
    stanford_sentiment, stanford_sentiment_1sent, stanford_sentiment_2sent, stanford_sentiment_3sent, \
    stanford_negation_features, stanford_negation_features_1sent, stanford_negation_features_2sent, stanford_negation_features_3sent, \
    stanford_based_verb_noun_sim, stanford_based_verb_noun_sim_1sent, stanford_based_verb_noun_sim_2sent, stanford_based_verb_noun_sim_3sent, \
    sdm_sim, stanford_avg_words_per_sent, stanford_avg_words_per_sent_1sent, stanford_avg_words_per_sent_2sent, stanford_avg_words_per_sent_3sent, \
    hedging_features, ppdb, discuss_features, single_flat_LSTM_50d_100, latent_dirichlet_allocation_300, NMF_cos_300
from fnc.refs.feature_engineering_benjamin import char_3grams_5000_concat_all_data, \
    lexical_features,max_diff_twitter_uni_bigrams,mpqa_unigrams, negated_context_word_12grams_concat_tf5000_l2_all_data, \
    nrc_emo_lex,nrc_hashtag_sentiment_unigram, nrc_hashtag_sentiment_unigram_POS, POS_features, readability_features , \
    sentiment140_unigrams, structural_features
from fnc.refs.utils.score import LABELS, score_submission
import csv
import fnc.refs.fnc1.scorer as scorer
from fnc.settings import myConstants

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Scorer pipeline')
    parser.add_argument('-p', '--pipeline_type', type=str, nargs='+', help='Pipeline Type (crossv,holdout,ftrain,ftest), e.g. -p crossv holdout', required=True)
    parser.add_argument('-s', '--scorer_type', type=str, help='Scorer Type (baselines, CNN, tf_idf, avg_embed, sdm, doc2vec, word_mover_sentence, word_mover_wholeText)', required=False)
    parser.add_argument('-t', '--threshold', type=float, help='Threshold', required=False)

    args = parser.parse_args()
    pipeline_type = args.pipeline_type
    scorer_type = args.scorer_type
    threshold = args.threshold
    return pipeline_type, scorer_type, threshold


def generate_features(stances, dataset, name, feature_list, features_dir):
    """
    Creates feature vectors out of the provided dataset
    """
    h, b, y, bodyId, headId = [], [], [], [], []

    feature_dict = {'overlap': word_overlap_features,
                    'refuting': refuting_features,
                    'polarity': polarity_features,
                    'hand': hand_features,
                    'word_unigrams_5000_concat_tf_l2_holdout_unlbled_test': word_unigrams_5000_concat_tf_l2_holdout_unlbled_test,
                    'NMF_cos_300_holdout_unlbled_test': NMF_cos_300_holdout_unlbled_test,
                    'NMF_concat_300_holdout_unlbled_test': NMF_concat_300_holdout_unlbled_test,
                    'latent_dirichlet_allocation_25_holdout_unlbled_test': latent_dirichlet_allocation_25_holdout_unlbled_test,
                    'latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test': latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test,
                    'NMF_fit_all_incl_holdout_and_test': NMF_fit_all_incl_holdout_and_test,
                    'latent_dirichlet_allocation_incl_holdout_and_test': latent_dirichlet_allocation_incl_holdout_and_test,
                    'latent_semantic_indexing_gensim_holdout_and_test': latent_semantic_indexing_gensim_holdout_and_test,
                    'NMF_fit_all_concat_300_and_test': NMF_fit_all_concat_300_and_test,
                    'word_ngrams_concat_tf5000_l2_w_holdout_and_test': word_ngrams_concat_tf5000_l2_w_holdout_and_test,
                    'NMF_fit_all': NMF_fit_all,
                    'word_ngrams_concat_tf5000_l2_w_holdout': word_ngrams_concat_tf5000_l2_w_holdout,
                    'latent_dirichlet_allocation': latent_dirichlet_allocation,
                    'latent_semantic_indexing_gensim_test': latent_semantic_indexing_gensim_test,
                    'NMF_fit_all_concat_300': NMF_fit_all_concat_300,
                    'NMF_cos_50': NMF_cos_50,
                    'latent_dirichlet_allocation_25': latent_dirichlet_allocation_25,
                    'latent_semantic_indexing_gensim_300_concat_holdout': latent_semantic_indexing_gensim_300_concat_holdout,
                    'NMF_concat_300_holdout': NMF_concat_300_holdout,
                    'word_unigrams_5000_concat_tf_l2_holdout': word_unigrams_5000_concat_tf_l2_holdout,
                    'ppdb': ppdb,
                    'stanford_ppdb': stanford_ppdb_score,
                    'stanford_ppdb_1sent': stanford_ppdb_score_1sent,
                    'stanford_ppdb_2sent': stanford_ppdb_score_2sent,
                    'stanford_ppdb_3sent': stanford_ppdb_score_3sent,
                    'stanford_sentiment': stanford_sentiment,
                    'stanford_sentiment_1sent': stanford_sentiment_1sent,
                    'stanford_sentiment_2sent': stanford_sentiment_2sent,
                    'stanford_sentiment_3sent': stanford_sentiment_3sent,
                    'stanford_wordsim': stanford_based_verb_noun_sim,
                    'stanford_wordsim_1sent': stanford_based_verb_noun_sim_1sent,
                    'stanford_wordsim_2sent': stanford_based_verb_noun_sim_2sent,
                    'stanford_wordsim_3sent': stanford_based_verb_noun_sim_3sent,
                    'stanford_negation': stanford_negation_features,
                    'stanford_negation_1sent': stanford_negation_features_1sent,
                    'stanford_negation_2sent': stanford_negation_features_2sent,
                    'stanford_negation_3sent': stanford_negation_features_3sent,
                    'stanford_avg_words_per_sent': stanford_avg_words_per_sent,
                    'stanford_avg_words_per_sent_1sent': stanford_avg_words_per_sent_1sent,
                    'stanford_avg_words_per_sent_2sent': stanford_avg_words_per_sent_2sent,
                    'stanford_avg_words_per_sent_3sent': stanford_avg_words_per_sent_3sent,
                    'hedging': hedging_features,
                    'sen2sen': sen2sen_similarity_max,
                    'wmdsenSen': word_mover_distance_similarity_sentence_min,
                    'wmdsenDoc': word_mover_distance_wholebody,
                    'sdm_sim': sdm_sim,
                    'discuss': discuss_features,
                    'single_flat_LSTM_50d_100': single_flat_LSTM_50d_100,
                    'char_3grams_5000_concat_all_data': char_3grams_5000_concat_all_data,
                    'lexical_features': lexical_features,
                    'max_diff_twitter_uni_bigrams': max_diff_twitter_uni_bigrams,
                    'mpqa_unigrams': mpqa_unigrams,
                    'negated_context_word_12grams_concat_tf5000_l2_all_data': negated_context_word_12grams_concat_tf5000_l2_all_data,
                    'nrc_emo_lex': nrc_emo_lex,
                    'nrc_hashtag_sentiment_unigram': nrc_hashtag_sentiment_unigram,
                    'nrc_hashtag_sentiment_unigram_POS': nrc_hashtag_sentiment_unigram_POS,
                    #'POS_features': POS_features,
                    'readability_features': readability_features,
                    'sentiment140_unigrams': sentiment140_unigrams,
                    'structural_features': structural_features,
                    'latent_dirichlet_allocation_300': latent_dirichlet_allocation_300,
                    'NMF_cos_300': NMF_cos_300
                    }

    stanceCounter = 0
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        bodyId.append(stance['Body ID'])
        headId.append(name+str(stanceCounter))
        stanceCounter += 1

    X_feat = []
    feat_list = []
    last_index = 0
    for feature in feature_list:
        feat = gen_or_load_feats(feature_dict[feature], h, b, features_dir+"/"+feature+"."+name+'.npy', bodyId, feature, headId, fold=name)
        feat_list.append((last_index, last_index+len(feat[0]), str(feature)))
        last_index += len(feat[0])
        X_feat.append(feat)
    X = np.concatenate(X_feat, axis=1)

    return X, y, feat_list

def generate_features_test(stances, dataset, name, feature_list, features_dir):
    """
    Equal to generate_features(), but creates features for the unlabeled test data
    """
    h, b, bodyId, headId = [], [], [], []

    feature_dict = {'overlap': word_overlap_features,
                    'refuting': refuting_features,
                    'polarity': polarity_features,
                    'hand': hand_features,
                    'word_unigrams_5000_concat_tf_l2_holdout_unlbled_test': word_unigrams_5000_concat_tf_l2_holdout_unlbled_test,
                    'NMF_cos_300_holdout_unlbled_test': NMF_cos_300_holdout_unlbled_test,
                    'NMF_concat_300_holdout_unlbled_test': NMF_concat_300_holdout_unlbled_test,
                    'latent_dirichlet_allocation_25_holdout_unlbled_test': latent_dirichlet_allocation_25_holdout_unlbled_test,
                    'latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test': latent_semantic_indexing_gensim_300_concat_holdout_unlbled_test,
                    'NMF_fit_all_incl_holdout_and_test': NMF_fit_all_incl_holdout_and_test,
                    'latent_dirichlet_allocation_incl_holdout_and_test': latent_dirichlet_allocation_incl_holdout_and_test,
                    'latent_semantic_indexing_gensim_holdout_and_test': latent_semantic_indexing_gensim_holdout_and_test,
                    'NMF_fit_all_concat_300_and_test': NMF_fit_all_concat_300_and_test,
                    'word_ngrams_concat_tf5000_l2_w_holdout_and_test': word_ngrams_concat_tf5000_l2_w_holdout_and_test,
                    'NMF_fit_all': NMF_fit_all,
                    'word_ngrams_concat_tf5000_l2_w_holdout': word_ngrams_concat_tf5000_l2_w_holdout,
                    'latent_dirichlet_allocation': latent_dirichlet_allocation,
                    'latent_semantic_indexing_gensim_test': latent_semantic_indexing_gensim_test,
                    'NMF_fit_all_concat_300': NMF_fit_all_concat_300,
                    'NMF_cos_50': NMF_cos_50,
                    'latent_dirichlet_allocation_25': latent_dirichlet_allocation_25,
                    'latent_semantic_indexing_gensim_300_concat_holdout': latent_semantic_indexing_gensim_300_concat_holdout,
                    'NMF_concat_300_holdout': NMF_concat_300_holdout,
                    'word_unigrams_5000_concat_tf_l2_holdout': word_unigrams_5000_concat_tf_l2_holdout,
                    'ppdb': ppdb,
                    'stanford_ppdb': stanford_ppdb_score,
                    'stanford_ppdb_1sent': stanford_ppdb_score_1sent,
                    'stanford_ppdb_2sent': stanford_ppdb_score_2sent,
                    'stanford_ppdb_3sent': stanford_ppdb_score_3sent,
                    'stanford_sentiment': stanford_sentiment,
                    'stanford_sentiment_1sent': stanford_sentiment_1sent,
                    'stanford_sentiment_2sent': stanford_sentiment_2sent,
                    'stanford_sentiment_3sent': stanford_sentiment_3sent,
                    'stanford_wordsim': stanford_based_verb_noun_sim,
                    'stanford_wordsim_1sent': stanford_based_verb_noun_sim_1sent,
                    'stanford_wordsim_2sent': stanford_based_verb_noun_sim_2sent,
                    'stanford_wordsim_3sent': stanford_based_verb_noun_sim_3sent,
                    'stanford_negation': stanford_negation_features,
                    'stanford_negation_1sent': stanford_negation_features_1sent,
                    'stanford_negation_2sent': stanford_negation_features_2sent,
                    'stanford_negation_3sent': stanford_negation_features_3sent,
                    'stanford_avg_words_per_sent': stanford_avg_words_per_sent,
                    'stanford_avg_words_per_sent_1sent': stanford_avg_words_per_sent_1sent,
                    'stanford_avg_words_per_sent_2sent': stanford_avg_words_per_sent_2sent,
                    'stanford_avg_words_per_sent_3sent': stanford_avg_words_per_sent_3sent,
                    'hedging': hedging_features,
                    'sen2sen': sen2sen_similarity_max,
                    'wmdsenSen': word_mover_distance_similarity_sentence_min,
                    'wmdsenDoc': word_mover_distance_wholebody,
                    'sdm_sim': sdm_sim,
                    'discuss': discuss_features,
                    'single_flat_LSTM_50d_100': single_flat_LSTM_50d_100,
                    'char_3grams_5000_concat_all_data': char_3grams_5000_concat_all_data,
                    'lexical_features': lexical_features,
                    'max_diff_twitter_uni_bigrams': max_diff_twitter_uni_bigrams,
                    'mpqa_unigrams': mpqa_unigrams,
                    'negated_context_word_12grams_concat_tf5000_l2_all_data': negated_context_word_12grams_concat_tf5000_l2_all_data,
                    'nrc_emo_lex': nrc_emo_lex,
                    'nrc_hashtag_sentiment_unigram': nrc_hashtag_sentiment_unigram,
                    'nrc_hashtag_sentiment_unigram_POS': nrc_hashtag_sentiment_unigram_POS,
                    #'POS_features': POS_features,
                    'readability_features': readability_features,
                    'sentiment140_unigrams': sentiment140_unigrams,
                    'structural_features': structural_features,
                    'latent_dirichlet_allocation_300': latent_dirichlet_allocation_300,
                    'NMF_cos_300': NMF_cos_300
                    }

    stanceCounter = 0
    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        bodyId.append(stance['Body ID'])
        headId.append(name+str(stanceCounter))
        stanceCounter += 1


    X_feat = []
    for feature in feature_list:
        print("calculate feature: " + str(feature))
        feat = gen_or_load_feats(feature_dict[feature], h, b, features_dir+"/"+feature+"_test."+name+'.npy', bodyId, feature, headId, fold=name)
        X_feat.append(feat)
        print(len(feat))
    X = np.concatenate(X_feat, axis=1)
    return X

def generate_non_bleeding_features(fold_stances, hold_out_stances, no_folds, BOW_feature_list, features_dir, d):
    """
    Does the same as generate_features(), just for non-bleeding features. It prevents bleeding by training certain features
    (e.g. word unigrams) strictly on the training data, instead of training+test data.
    Feature extraction methods in feature_engineering.py have to provide an extended parameter list in order to use this
    (method_name(headlines, bodies, headlines_test, bodies_test)). The saved feature files have the following structure:
        - e.g. feature_name_0.py will hold the features of the folds from 1 to 9 and feature_name_0.test.py
            will hold the features of fold 0, derived of the folds 1 to 9

    This method (and feature methods based on this one) is just to get more reliable (non-bleeding) score results and cannot be used for
    the training of the final classifier.
    """

    #data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
    #d = DataSet(data_path)

    # holds all bag of words features and their feature extraction methods
    non_bleeding_feature_dict = {}

    def generate_holdout_BOW_features():
        for fold in range(no_folds):
            ids = list(range(no_folds))
        merged = []
        for i in ids:
            merged.extend(fold_stances[i])

        h = []
        b = []
        for stance in merged:
            h.append(stance['Headline'])
            b.append(d.articles[stance['Body ID']])

        h_test = []
        b_test = []
        for stance in hold_out_stances:
            h_test.append(stance['Headline'])
            b_test.append(d.articles[stance['Body ID']])

        gen_non_bleeding_feats(non_bleeding_feature_dict[feature], h, b, h_test, b_test,
                               features_dir, feature, 'holdout')

    def generate_fold_BOW_features():
        for fold in range(no_folds):
            ids = list(range(no_folds))
            del ids[fold]

            merged = []
            for i in ids:
                merged.extend(fold_stances[i])

            # 9 of 10 folds merged for training BOW features
            h = []
            b = []
            for stance in merged:
                h.append(stance['Headline'])
                b.append(d.articles[stance['Body ID']])

            # 1 fold (test) to extract features out of the generated BOW
            h_test = []
            b_test = []
            for stance in fold_stances[fold]:
                h_test.append(stance['Headline'])
                b_test.append(d.articles[stance['Body ID']])

            gen_non_bleeding_feats(non_bleeding_feature_dict[feature],
                                   h, b, h_test, b_test, features_dir, feature,
                                   fold)

    for feature in BOW_feature_list:
        generate_fold_BOW_features()
        generate_holdout_BOW_features()


def concat_non_bleeding_features(X_train, X_test, BOW_feature_list, features_dir, fold):
    """
    Concatenates the given train and test feature vectors with all the non bleeding features
    specified in the non_bleeding_feature_list.
    """
    feat_list = []
    for feature in BOW_feature_list:
        X_train_part = np.load("%s/%s.%s.npy" % (features_dir, feature, fold))
        last_index = len(X_train[0])
        X_train = np.concatenate([X_train, X_train_part], axis=1)
        feat_list.append((last_index, last_index+len(X_train_part[0]), str(feature)))
        X_test_part = np.load("%s/%s.%s.test.npy" % (features_dir, feature, fold))
        X_test = np.concatenate([X_test, X_test_part], axis=1)
    return X_train, X_test, feat_list

def print_score_from_restored_model(clf, X_test, y_test):
    """
    Takes a fitted classifier, predicts on base on the given X,
    compares to the actual y and prints the score.
    """
    y_predicted = clf.predict(X_test)
    predicted = [LABELS[int(a)] for a in y_predicted]
    actual = [LABELS[int(a)] for a in y_test]

    # calc FNC score
    fold_score, _ = score_submission(actual, predicted)
    max_fold_score, _ = score_submission(actual, actual)
    score = fold_score / max_fold_score

    print("FNC-1 score from restored model: " +  str(score) +"\n")

    return score

def save_model(clf, save_folder, filename):
    """
    Dumps a given classifier to the specific folder with the given name
    """
    import pickle
    path = save_folder + filename
    with open(path, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(save_folder, filename):
    """
    Loads and returns a classifier at the given folder with the given name
    """
    print("Warning: Make sure older models with this name have been trained on the same features! Otherwise,"
          "if the lengths of the features the model has been trained on, differ, an error will occur!")
    import pickle
    path = save_folder + filename
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def get_save_folder(parent_folder, scorer_type):
    """
    Returns an unused save location for a classifier based on its name
    """
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # in order to prevent overwriting existing learned models, always create a new folder
    save_folder = parent_folder + scorer_type
    id = 0
    while os.path.exists(save_folder+"_"+str(id)):
        id += 1
    save_folder += "_" + str(id) +"/"
    os.makedirs(save_folder)
    return save_folder

def cross_validation(fold_stances, folds, Xs, ys, non_bleeding_features, features_dir,
                     scorer_type, all_accuracies_related, all_accuracies_stance,
                     all_f1_related, all_f1_stance, all_scores, result_string, learning_rate_string):
    best_score = 0

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        # Add BOW features to current feature vectors
        # The features are specified in BOW_feature_list
        X_train, X_test, _ = concat_non_bleeding_features(
            X_train, X_test,
            non_bleeding_features, features_dir, fold)

        # get the estimator for this loop
        clf = esitmator_definitions.get_estimator(scorer_type)

        print("Begin fitting at: " + str(datetime.datetime.now()).split('.')[0] + "\n")

        # start fitting the estimator
        clf.fit(X_train, y_train)

        # predict the labes for fitted classifier with the test data
        predicted_int = clf.predict(X_test)

        #Baseline "hack"
        # ToDO: Remove after baseline test
        predicted_int = np.empty(len(y_test))
        predicted_int.fill(3)

        predicted = [LABELS[int(a)] for a in predicted_int]
        actual = [LABELS[int(a)] for a in y_test]

        # calculate the FNC-1 score based on the predicted and the actual labels
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)
        score = fold_score / max_fold_score

        # calculates accuracy and f1-macro scores
        accuracy_stance = score_calculation.get_accuracy(predicted_int, y_test, stance=True)
        accuracy_related = score_calculation.get_accuracy(predicted_int, y_test, stance=False)
        f1_stance = score_calculation.get_f1score(predicted_int, y_test, stance=True)
        f1_related = score_calculation.get_f1score(predicted_int, y_test, stance=False)

        # add the scores to the list holding the stores of all folds
        all_accuracies_related.append(accuracy_related)
        all_accuracies_stance.append(accuracy_stance)
        all_f1_related.append(f1_related)
        all_f1_stance.append(f1_stance)

        # get best score of all folds
        all_scores.append(score)
        if score > best_score:
            best_score = score

        # Prepare printout for fold result
        printout = printout_manager.get_foldwise_printout(fold, accuracy_related, accuracy_stance, f1_related,
                                                          f1_stance, score)
        print(printout)  # print results for this fold
        result_string += printout  # add results to final result file

        # add to special file that shows learning rate and loss of optimizer
        if isinstance(clf, MultiThreadingFeedForwardMLP):
            learning_rate_string += clf.get_learning_rates(fold) + "\n"

    # Prepare printout for final result
    printout = printout_manager.get_cross_validation_printout(
        all_accuracies_related, all_accuracies_stance, all_f1_related, all_f1_stance, all_scores, best_score)
    print(printout)  # print cross validation results
    result_string += printout  # add cross validation results to result file

    return result_string, learning_rate_string

#Taken from Benjamins LSTM
def append_to_loss_monitor_file(text, filepath):
    with open(filepath, 'a+') as the_file:
        the_file.write(text+"\n")


def validate_holdout(Xs, ys, X_holdout, y_holdout, non_bleeding_features, features_dir,
                     scorer_type, feat_indices, result_string, learning_rate_string, features):
    """
    Trains the classifier on all of the train+test data and tests it on the holdout set
    :param Xs: All the training data's feature vectors, split in their folds
    :param ys: All the training data's labels, split in their folds
    :param X_holdout: The holdout feature vectors
    :param y_holdout: The holdout labels
    :param non_bleeding_features: The list of non-bleeding features that has to be concatenated to the existing feature vectors
    :param features_dir: the directory where the features are stored
    :param scorer_type: the scorer type, e.g. MLB_base (see estimator_definitions.py in utils folder)
    :param feat_indices: indices returned by generate_features() method. They indicate at what index of the feature vector a specific
    feature starts and where it ends. This is used for printing out the feature importances by the RandomForest classifier
    :param result_string: The current result string in order to add the holdout results
    :param learning_rate_string: The current learning rate string in order to add information about the learning rate
    :return: the updated result_string and learning_rate_string
    """
    # define folder to save the classifier and create it if not existing
    parent_folder = "%s/data/fnc-1/mlp_models/" % (path.dirname(path.dirname(path.abspath(__file__))))

    # create the new save folder
    save_folder = get_save_folder(parent_folder, scorer_type+"_new")

    # only pass a save folder if the classifier should be saved
    best_clf = esitmator_definitions.get_estimator(scorer_type, save_folder=save_folder)

    # stack all the feature vectors of all the folds
    X_train = np.vstack(tuple([Xs[i] for i in range(10)]))
    y_train = np.hstack(tuple([ys[i] for i in range(10)]))

    # concat non-bleeding features
    X_train, X_holdout, feat_indices_holdout = concat_non_bleeding_features(
        X_train, X_holdout,
        non_bleeding_features, features_dir, 'holdout')

    # test for oversampling: fits the current classifier, oversampled with a given
    # method and checks the score on the holdout set
    use_over_sampling = False
    if use_over_sampling == True:
        from imblearn.over_sampling import SMOTE
        kind = ['regular', 'borderline1', 'borderline2', 'svm']
        for m in kind:
            sm = SMOTE(kind=m)
            X_res, y_res = sm.fit_sample(X_train, y_train)
            best_clf.fit(X_res, y_res)
            y_predicted = best_clf.predict(X_holdout)
            predicted = [LABELS[int(a)] for a in y_predicted]
            actual = [LABELS[int(a)] for a in y_holdout]
            fold_score, _ = score_submission(actual, predicted)
            max_fold_score, _ = score_submission(actual, actual)
            score = fold_score / max_fold_score
            print("Score " + m +  ":" + str(score))


    #Taken from Benjamins LSTM
    loss_monitor_file_dir = "%s/data/fnc-1/model_results/loss_results/" % (
        path.dirname(path.dirname(path.abspath(__file__))))
    loss_filename = loss_monitor_file_dir + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + ".txt"
    # fit the classifier
    if 'f_ext' in scorer_type:
        append_to_loss_monitor_file("\n\nFOLD holdout and classifier: " + scorer_type + "\n", loss_filename)
        append_to_loss_monitor_file(str(datetime.datetime.now()).split('.')[0], loss_filename)
        best_clf.fit(X_train, y_train, X_holdout, np.array(y_holdout), 'holdout', loss_filename)
    else:
        best_clf.fit(X_train, y_train)

    # predict labels
    y_predicted = best_clf.predict(X_holdout)
    predicted = [LABELS[int(a)] for a in y_predicted]
    actual = [LABELS[int(a)] for a in y_holdout]

    # calc FNC score
    fold_score, cm = score_submission(actual, predicted)
    max_fold_score, _ = score_submission(actual, actual)
    score = fold_score / max_fold_score

    # calc accuracy for related/unrelated and stances
    accuracy_stance = score_calculation.get_accuracy(y_predicted, y_holdout, stance=True)
    accuracy_related = score_calculation.get_accuracy(y_predicted, y_holdout, stance=False)
    f1_stance = score_calculation.get_f1score(y_predicted, y_holdout, stance=True)
    f1_related = score_calculation.get_f1score(y_predicted, y_holdout, stance=False)

    # prepare printout for final results of holdout set
    printout = printout_manager.get_holdout_printout(save_folder, accuracy_related, accuracy_stance, f1_related, f1_stance, score)
    printout += printout_manager.calculate_confusion_matrix(cm)
    print(printout) # print holdout results
    result_string += printout + "\n"# add results to string that is going to be saved into a file

    result_file_folder = "%s" % (path.dirname(path.dirname(path.abspath(__file__))))
    printout_manager.save_file(result_string, result_file_folder + "/fnc_results_holdout.txt", "a+")

    #aligned printout for ablation:
    summary = printout_manager.get_holdout_ablation_printout(features, score,f1_stance,save_folder)
    printout_manager.save_file(summary, result_file_folder + "/fnc_results_holdout_summary.txt", "a+")

    # test saving and restoring model
    #filename = scorer_type + ".sav"
    #save_model(best_clf, save_folder,filename)
    #load_clf = load_model(parent_folder + scorer_type + "_new_0/", filename) # the 0th folder should always exist
    #print_score_from_restored_model(load_clf, X_holdout, y_holdout)

    # add to special file that shows learning rate and loss of optimizer
    if isinstance(best_clf, MultiThreadingFeedForwardMLP):
        learning_rate_string += best_clf.get_learning_rates('holdout') + "\n"

    # print feature importances
    if scorer_type == 'randomforest':
        result_file_folder = "%s" % (path.dirname(path.dirname(path.abspath(__file__))))
        importances = best_clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in best_clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        feat_indices.append(feat_indices_holdout)

        feat_importance_string = str(feat_indices) + "\n"
        for i in indices:
            feat_importance_string += str(i) + ";" + str(importances[i]) + ";" + str(std[i]) + "\n"

        # save feature importances as file
        printout_manager.save_file(feat_importance_string, result_file_folder + "/feat_importance_rf.txt", "a+")

    return result_string, learning_rate_string


def final_clf_training(Xs, ys, X_holdout, y_holdout, scorer_type, sanity_check=False, oversampling=False):
    """
    Train final classifier on all of the data to prepare it for the prediction of the FNC-1's unlabeled data
    :param Xs: All the training data's feature vectors, split in their folds
    :param ys: All the training data's labels, split in their folds
    :param X_holdout: The holdout feature vectors
    :param y_holdout: The holdout labels
    :param scorer_type: the scorer type, e.g. MLB_base (see estimator_definitions.py in utils folder)
    :param sanity_check: If true, the trained classifier predicts the labels of the data it was trained on and prints out the score
    :return: the final classifier
    """

    # stack all the feature vectors of all the folds
    X_train = np.vstack(tuple([Xs[i] for i in range(10)]))
    y_train = np.hstack(tuple([ys[i] for i in range(10)]))

    # stack the holdout feature vectors on the feature vectors of all folds
    X_all = np.concatenate([X_train, X_holdout], axis=0)
    y_all = np.concatenate([y_train, y_holdout], axis=0)

    # define and create parent folder to save all trained classifiers into
    parent_folder = "%s/data/fnc-1/mlp_models/" % (path.dirname(path.dirname(path.abspath(__file__))))

    # create the new save folder for the specific classifer
    scorer_folder_name = scorer_type+"_final"
    save_folder = get_save_folder(parent_folder, scorer_folder_name+"_new")

    # get classifier and only pass a save folder if the classifier should be saved
    clf = esitmator_definitions.get_estimator(scorer_type, save_folder=save_folder)

    #perform oversampling if selected
    if oversampling == True:
        if 'f_ext' in scorer_type:
            print("Oversampling not defined for LSTM")
            exit()

        import datetime
        start = datetime.datetime.now().time()
        print("Started oversampling/undersampling at: " + str(start))
        ######
        # Oversampling

        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        print("Oversampling data")
        #kind = ['regular', 'borderline1', 'borderline2', 'svm']
        #sm = SMOTE(kind='regular',)
        #X_res, y_res = sm.fit_sample(X_all, y_all)

        #ros = RandomOverSampler()
        #X_res, y_res = ros.fit_sample(X_all, y_all)

        #ada = ADASYN()
        #X_res, y_res = ada.fit_sample(X_all, y_all)

        ######################################################
        # Undersampling
        from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, CondensedNearestNeighbour, \
            NeighbourhoodCleaningRule, InstanceHardnessThreshold
        # remove Tomek links
        tl = TomekLinks(return_indices=True)
        X_res, y_res, idx_resampled = tl.fit_sample(X_all, y_all)

        #enn = EditedNearestNeighbours(random_state=0)
        #X_res, y_res = enn.fit_sample(X_all, y_all)

        #cnn = CondensedNearestNeighbour(random_state=0)
        #X_res, y_res = cnn.fit_sample(X_all, y_all)

        #ncr = NeighbourhoodCleaningRule(random_state=0)
        #X_res, y_res = ncr.fit_sample(X_all, y_all)

        #iht = InstanceHardnessThreshold(random_state=0, estimator=clf)
        #X_res, y_res = iht.fit_sample(X_all, y_all)


        ##################
        # Combination of Undersampling and oversampling

        from imblearn.combine import SMOTEENN, SMOTETomek
        #smote_enn = SMOTEENN(random_state=0)
        #X_res, y_res = smote_enn.fit_sample(X_all, y_all)

        #smote_tomek = SMOTETomek(random_state=0)
        #X_res, y_res = smote_tomek.fit_sample(X_all, y_all)

        end = datetime.datetime.now().time()
        print("Ended oversampling/undersampling at: " + str(end))

        clf.fit(X_res, y_res)
    else: # if oversampling is false
        import datetime
        # fit the final classifier
        loss_monitor_file_dir = "%s/data/fnc-1/model_results/loss_results/" % (
            path.dirname(path.dirname(path.abspath(__file__))))
        loss_filename = loss_monitor_file_dir + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + ".txt"
        # fit the classifier
        if 'f_ext' in scorer_type:
            append_to_loss_monitor_file("\n\nFOLD holdout and classifier: " + scorer_type + "\n", loss_filename)
            append_to_loss_monitor_file(str(datetime.datetime.now()).split('.')[0], loss_filename)
            clf.fit(X_train, y_train, X_holdout, np.array(y_holdout), 'holdout', loss_filename)
        else:
            clf.fit(X_all, y_all)

    # save the model
    filename = scorer_folder_name + ".sav"
    save_model(clf, save_folder, filename)  # save model with filename to specific folder

    # predict on the data the classifier was trained on => should give near perfect score
    if sanity_check == True:
        # get predicted and actual labels
        y_predicted = clf.predict(X_all)
        predicted = [LABELS[int(a)] for a in y_predicted]
        actual = [LABELS[int(a)] for a in y_all]

        # calc FNC score
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)
        score = fold_score / max_fold_score

        # calc accuracy, f1 macro
        accuracy_stance = score_calculation.get_accuracy(y_predicted, y_all, stance=True)
        accuracy_related = score_calculation.get_accuracy(y_predicted, y_all, stance=False)
        f1_stance = score_calculation.get_f1score(y_predicted, y_all, stance=True)
        f1_related = score_calculation.get_f1score(y_predicted, y_all, stance=False)

        # printout results
        printout = printout_manager.get_holdout_printout(save_folder, accuracy_related, accuracy_stance, f1_related,
                                                         f1_stance, score)
        print("SANITY CHECK (predict on train data):")
        print(printout)
    return clf

def final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, final_clf):
    """
    Run the prediction on the final model. In order to do that, the features vectors of the unlabeled FNC-1 data are
    generated first.
    :param data_path: data_path to the unlabeled stances and the corresponding bodies
    :param features: The feature list
    :param features_dir: The directory where the features are stored
    :param scorer_type: the scorer type, e.g. MLB_base (see estimator_definitions.py in utils folder)
    :param run_final_train: Sanity check: if the final classifier has been trained in this run, check if the prediction of it
    compared to the classifier that is being loaded in this method, are the same. If yes, they represent the same model.
    :param final_clf: The classifier that was trained in this run (IF a classifier was trained)
    :return:
    """

    d = myConstants.testdataset

    # generate features for the unlabeled testing set
    X_final_test = generate_features_test(d.stances, d, str("final_test"), features, features_dir)

    # define and create parent folder to save all trained classifiers into
    parent_folder = "%s/data/fnc-1/mlp_models/" % (path.dirname(path.dirname(path.abspath(__file__))))
    fnc_result_folder = "%s/data/fnc-1/fnc_results/" % (path.dirname(path.dirname(path.abspath(__file__))))

    # load model [scorer_type]_final_2 classifier
    filename = scorer_type + "_final.sav"
    load_clf = load_model(parent_folder + scorer_type + myConstants.model_name, filename)
    # The model is set in settings.py in class "myConstants"

    print("Load model for final prediction of test set: " + parent_folder + scorer_type + myConstants.model_name + filename)

    # predict classes and turn into labels
    y_predicted = load_clf.predict(X_final_test)
    predicted = [LABELS[int(a)] for a in y_predicted]

    # create folder to save the file
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    if not os.path.exists(fnc_result_folder):
        os.makedirs(fnc_result_folder)

    # save the submission file, including the prediction for the labels
    with open(fnc_result_folder + scorer_type + "_submission.csv", 'w') as csvfile:
        fieldnames = ["Headline", "Body ID", "Stance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i = 0
        for stance in d.stances:
            writer.writerow(
                {'Headline': stance['Headline'], 'Body ID': stance['Body ID'], 'Stance': predicted[i]})
            i += 1


    # save the probabilities file, including the prediction for the labels
    if ("voting_" not in scorer_type) and ("f_ext" not in scorer_type) and ("MLP_base_2" not in scorer_type):
        print("Generating submission_probabilities.csv")
        predicted_proba = load_clf.predict_proba(X_final_test)
        with open(fnc_result_folder + scorer_type + "_probabilities.csv", 'w') as csvfile:
            fieldnames = ["Headline", "Body ID", "Agree", "Disagree", "Discuss", "Unrelated"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            i = 0
            for stance in d.stances:
                writer.writerow(
                    {'Headline': stance['Headline'], 'Body ID': stance['Body ID'], 'Agree': predicted_proba[i][0],
                     'Disagree': predicted_proba[i][1], 'Discuss': predicted_proba[i][2],
                     'Unrelated': predicted_proba[i][3]})
                i += 1
    else:
        print("Not generating submission_probabilities.csv - because classifier contains \"voting\", \"f_ext\" or \"MLP_base_2\"")

    # check whether loaded clf from disk and just trained clf return the same results
    if (run_final_train == True) and not (final_clf is None):
        print("Check whether loaded final model and just trained final model retrieve the same results."
              "The results are only equal (=None) if they are the same model. This is a check to see whether "
              "saving and loading the model works correctly:")
        print(np.testing.assert_array_equal(y_predicted, final_clf.predict(X_final_test)))

def delete_ffmlp_data():
    """
    In order to copy the structure of Sklearn's BaseEstimator (fit(), predict(), ...) the MultiThreadingFeedForwardMLP
    has to save its graph after fitting. If its argument "save_folder" doesn't get a specific folder, it's seen as a
    temporary model (lifetime of the model is just for the runtime). The model will be saved in a special temporary folder.
    This method is called after the pipeline run has finished and deletes all the temporarily saved models of
    MultiThreadingFeedForwardMLP.
    """
    import shutil
    ffmlp_dir = "%s/data/fnc-1/mlp_models/temp_models" % (
    path.dirname(path.dirname(path.abspath(__file__))))
    if (os.path.exists(ffmlp_dir)):
        for the_file in os.listdir(ffmlp_dir):
            file_path = os.path.join(ffmlp_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def pipeline():
    # define data paths
    result_file_folder = "%s" % (path.dirname(path.dirname(path.abspath(__file__))))
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
    embeddPath = "%s/data/embeddings/google_news/GoogleNews-vectors-negative300.bin.gz" % (path.dirname(path.dirname(path.abspath(__file__))))

    # get arguments for pipeline call
    pipeline_type, scorer_type, threshold = get_args()
    splits_dir = myConstants.splits_dir
    features_dir = myConstants.features_dir

    # configure pipeline runs by using given terminal arguments
    run_CV = False
    if "crossv" in pipeline_type:
        run_CV = True  # run cross validation

    run_validation = False
    if "holdout" in pipeline_type:
        run_validation = True  # run validation on holdout set

    run_final_train = False
    if "ftrain" in pipeline_type:
        run_final_train = True  # train classifier on all the data available

    run_final_prediction = False
    if "ftest" in pipeline_type:
        run_final_prediction = True  # run prediction on test data provided by FNC-1 challenge

    if "analysis" in pipeline_type:
        # parse in train bodies and stances for analysis
        bodyDict = myConstants.testdataset.articles
        train_data = myConstants.testdataset.stances

        if scorer_type == None:
            raise AttributeError("Please specify scorer_type")
        if threshold == None:
            raise AttributeError("Please specify Threshold")
        model = Model(scorer_type, embeddPath=embeddPath)
        model.analyze_data(train_data, bodyDict, threshold=threshold)

    perform_oversampling = myConstants.perform_oversampling

    # train the model / predict on basis of the model
    if True in [run_CV, run_validation, run_final_train, run_final_prediction]:

        if sys.version_info.major < 3:
            sys.stderr.write('Please use Python version 3 and above\n')
            sys.exit(1)

        d = myConstants.d

        folds, hold_out = kfold_split(d, n_folds=10, base_dir=splits_dir)
        fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

        Xs = dict()
        ys = dict()

        # (scorer_type, [normal features], [non-bleeding features])
        feature_list = [
            # ORIGINAL FEATURES OF FNC-1 BEST SUBMISSION 3)
            ('voting_mlps_hard',
             ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
              'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
              'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
              'stanford_wordsim_1sent'],
             [])
        ]

        feature_list = myConstants.feature_list

        for scorer_type, features, non_bleeding_features in feature_list:

            # print classifier and features for this loop
            print(scorer_type)
            print(features)
            print(non_bleeding_features)

            # generate bow features for later use
            generate_non_bleeding_features(fold_stances, hold_out_stances, 10, non_bleeding_features, features_dir, d)

            # Load/Precompute all features now
            X_holdout, y_holdout, feat_indices = generate_features(hold_out_stances, d, "holdout", features, features_dir)
            for fold in fold_stances:
                Xs[fold], ys[fold], _ = generate_features(fold_stances[fold], d, str(fold), features, features_dir)

            # initialize lists needed to save results for each fold
            all_accuracies_related = []
            all_accuracies_stance = []
            all_f1_related = []
            all_f1_stance = []
            all_scores = []

            # get head for result file
            file_head = printout_manager.get_printout_file_head(scorer_type, features, non_bleeding_features)
            result_string = file_head # use head for result file
            learning_rate_string = file_head # use head for learning rate file

            # run cross validation on the specified folds
            if run_CV == True:
                result_string, learning_rate_string = cross_validation(fold_stances, folds, Xs, ys, non_bleeding_features, features_dir,
                                 scorer_type, all_accuracies_related, all_accuracies_stance,
                                 all_f1_related, all_f1_stance, all_scores, result_string, learning_rate_string)

            # Test on holdout set
            if run_validation == True:
                result_string, learning_rate_string = validate_holdout(Xs, ys, X_holdout, y_holdout, non_bleeding_features,
                                 features_dir, scorer_type, feat_indices, result_string, learning_rate_string, features)

            # Train the final classifer
            if run_final_train == True:
                final_clf = final_clf_training(Xs, ys, X_holdout, y_holdout, scorer_type, sanity_check=True, oversampling=perform_oversampling)

            # Run the final classifier on the test data
            if run_final_prediction == True:
                if run_final_train == True:
                    final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, final_clf)
                else:
                    final_clf_prediction(data_path, features, features_dir, scorer_type, run_final_train, None)

                # calculate FNC Score
                fnc_result_folder = "%s/data/fnc-1/fnc_results/%s_submission.csv" % (path.dirname(path.dirname(path.abspath(__file__))),  scorer_type)
                fnc_gold_labels_file = myConstants.test_stances_gold
                predicted_set = scorer.load_dataset(fnc_result_folder)
                fnc_gold_labels = scorer.load_dataset(fnc_gold_labels_file)
                test_score, cm, f1_score = scorer.score_submission(fnc_gold_labels, predicted_set)
                null_score, max_score = scorer.score_defaults(fnc_gold_labels)

                fnc_results = "################################################ \n"
                fnc_results += "Corpora: " + myConstants.datasetName + "\n"
                fnc_results += "Model:" + scorer_type + myConstants.model_name + "\n"
                if perform_oversampling == True:
                    fnc_results += "Using oversampling \n"
                fnc_results += result_string + "\n" + printout_manager.calculate_confusion_matrix(cm)
                fnc_results += scorer.SCORE_REPORT.format(max_score, null_score, test_score) + "\n"
                fnc_results += "\nRelative FNC Score: {:.3f}".format(100/max_score*test_score) + "% \n"
                fnc_results += "\n" + f1_score + "\n"

                print(fnc_results)
                printout_manager.save_file(fnc_results, result_file_folder + "/fnc_results.txt", "a+")
                #scorer.print_confusion_matrix(cm)
                #print(scorer.SCORE_REPORT.format(max_score, null_score, test_score))

            # save file with results to disk
            printout_manager.save_file(result_string, result_file_folder + "/result_file_temp.txt", "a+")

            # save file with learning rates to disk
            learning_rate_string += "===================================\n"
            printout_manager.save_file(learning_rate_string, result_file_folder + "/learning_rate_file_temp.txt", "a+")

            # delete temporary saved MultiThreadingFeedForwardMLP models if existing
            delete_ffmlp_data()

if __name__ == '__main__':
    pipeline()
