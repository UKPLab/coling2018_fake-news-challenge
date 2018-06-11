from keras import backend as K
from fnc.refs.feature_engineering_helper import misc

import os
INCORRECT_PRED_PATH = "%s/data/claim_validation/error_analysis/incorrect_predictions/" % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FEATURES_DIR = "%s/data/claim_validation/features/" % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def append_to_loss_monitor_file(text, filepath):
    with open(filepath, 'a+') as the_file:
        the_file.write(text+"\n")

def f1_macro(y_true, y_pred):
    """
    DOESN'T WORK CORRECTLY
    Can be used as metric in model.compile(metrics=[f1__macro],...).
    This is only approximated because it's calculated after each batch and then averaged.
    It's worst in the first few epochs.
    #https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
    #https://github.com/fchollet/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py
    #https://gist.github.com/Mistobaan/337222ac3acbfc00bdac7
    #https://github.com/fchollet/keras/blob/ac1a09c787b3968b277e577a3709cd3b6c931aa5/tests/keras/test_metrics.py
    #https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
    #https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    :param y_true:
    :param y_pred:
    :return:
    """

    CLASSES = 2

    def recall(y_true, y_pred, class_id):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_true, class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    def prec(y_true, y_pred, class_id):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    r = 0.0
    p = 0.0
    for class_id in range(CLASSES):
        r += recall(y_true, y_pred, class_id)
        p += prec(y_true, y_pred, class_id)

    r /= CLASSES
    p /= CLASSES

    return (2 * p * r) / (p + r)

def convert_data_to_one_hot(y_train):
    # y_test_temp = np.zeros((y_test.size, y_test.max() + 1), dtype=np.int)
    # y_test_temp[np.arange(y_test.size), y_test] = 1

    # Other option:
    #   y_train is a tensor then because of one_hot, but feed_dict only accepts numpy arrays => replace y_train with sess.run(y_train)
    #   http://stackoverflow.com/questions/34410654/tensorflow-valueerror-setting-an-array-element-with-a-sequence
    # return tf.one_hot(y_train, 4), tf.one_hot(y_test, 4)
    y_train_temp = np.zeros((y_train.size, y_train.max() + 1), dtype=np.int)
    y_train_temp[np.arange(y_train.size), y_train] = 1

    return y_train_temp

def split_X(X_train, MAX_SEQ_LENGTH_HEADS):
    # split to get [heads, docs]
    X_train_splits = np.hsplit(X_train, np.array([MAX_SEQ_LENGTH_HEADS]))
    X_train_head = X_train_splits[0]
    X_train_doc = X_train_splits[1]

    print("X_train_head.shape = " + str(np.array(X_train_head).shape))
    print("X_train_doc.shape = " + str(np.array(X_train_doc).shape))

    return X_train_head,X_train_doc

def save_incorrectly_predicted(y_pred, y_true, fold, MODELNAME):
    """
    Saves the samples that are incorrectly predicted into a separate file for false and true
    and also differentiates by fold
    :param y_pred:
    :param y_true:
    :param fold:
    :param PATH:
    :param MODELNAME:
    :return:
    """
    """INCORRECT_PRED_PATH = "%s/data/claim_validation/error_analysis/incorrect_predictions/" % (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    FEATURES_DIR = "%s/data/claim_validation/features/" % (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    y_pred = [1, 0, 0, 0, 1, 1]
    y_true = [0, 0, 0, 1, 0, 1]
    fold = 'holdout'
    MODELNAME = 'h_biLSTM'
    """

    false_positive = []
    false_negative =[]

    for i in range(len(y_pred)): #LABELS_2 = ['true', 'false']
        if y_pred[i] == 0 and y_true[i] == 1:
            false_positive.append(i)
        elif y_pred[i] == 1 and y_true[i] == 0:
            false_negative.append(i)

    PATH = INCORRECT_PRED_PATH + MODELNAME + "/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)


    with open(PATH + "false_positive."+str(fold)+".csv", "a+") as csvfile:
        file = ""
        for index in false_positive:
            file += str(index) + ";"
        csvfile.write(file+"\n")

    with open(PATH + "false_negative."+str(fold)+".csv", "a+") as csvfile:
        file = ""
        for index in false_negative:
            file += str(index) + ";"
        csvfile.write(file+"\n")


"""def get_repeatedly_misqualified_samples(fold, MODELNAME, FEATURENAME, save_file=True, with_unknown = True):
    \"""
    Returns a list of false negative and false positive samples. Also saves the text of all
    these samples to the INCORRECT_PRED_PATH/MODELNAME folder.
    :param fold:
    :param MODELNAME:
    :param FEATURENAME:
    :param save_file:
    :return:
    \"""
    import csv
    import pickle

    def reconstruct_string(X, vocab):
        reconstructed_dict = {}
        vocab_revert = {y: x for x, y in vocab.items()}
        for i, claim in enumerate(X):
            temp_string = ""
            for index in claim:
                if index != 0:
                    temp_string += vocab_revert.get(index, "") + " "
            reconstructed_dict[i] = temp_string
        return reconstructed_dict

    def get_intersection_result(dictionary):
        result_set = set
        for key, value in dictionary.items():
            if key == 0:
                result_set = set(value)
            else:
                result_set = result_set.intersection(set(value))
        return result_set

    PATH = INCORRECT_PRED_PATH + MODELNAME + "/"

    false_positive = {}
    false_negative = {}

    with open(PATH + "false_positive."+str(fold)+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            temp = []
            for index in row:
                if index != "":
                    temp.append(int(index))
            false_positive[i] = temp

    with open(PATH + "false_negative."+str(fold)+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            temp = []
            for index in row:
                if index != "":
                    temp.append(int(index))
            false_negative[i] = temp

    false_positive_intersection = get_intersection_result(false_positive)
    false_negative_intersection = get_intersection_result(false_negative)


    # flat bisltm features do not save the string sequences, so they have to be reconstructed from the vocab
    if "LSTM_100d_GloVe_400k_1000_250_50_non_sparse" in FEATURENAME:
            #and not os.path.exists(FEATURES_DIR+FEATURENAME+"_claim_string_seq_dict"+ "." + str(fold) + ".pkl")\
        with open(FEATURES_DIR + FEATURENAME+"_param_dict.pkl", "rb") as f:
            param_dict = pickle.load(f)

        # load X data
        X_data = np.load(FEATURES_DIR + FEATURENAME+"."+fold+".npy")

        X_data_claims, X_data_orig_docs, X_data_evid = split_X(X_data, param_dict["MAX_SEQ_LENGTH_CLAIMS"],
                                                                  param_dict["MAX_SEQ_LENGTH_DOCS"])

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['CLAIM_VOCAB_FILE'], 'rb') as f:
            CLAIM_VOCAB_FILE = pickle.load(f)
        claim_text_full = reconstruct_string(X_data_claims, CLAIM_VOCAB_FILE)


        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['ORIG_DOC_VOCAB_FILE'], 'rb') as f:
            ORIG_DOC_VOCAB_FILE = pickle.load(f)
        orig_doc_text_full = reconstruct_string(X_data_orig_docs, ORIG_DOC_VOCAB_FILE)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['EVIDENCE_VOCAB_FILE'], 'rb') as f:
            EVIDENCE_VOCAB_FILE = pickle.load(f)
        evid_text_full = reconstruct_string(X_data_evid, EVIDENCE_VOCAB_FILE)

        # add new sequence dict entries to param dict
        param_dict['ORIG_DOC_STRING_SEQUENCES_DICT'] = FEATURENAME+"_claim_string_seq_dict"
        param_dict['CLAIM_STRING_SEQUENCES_DICT'] = FEATURENAME+"_orig_doc_string_seq_dict"
        param_dict['EVID_STRING_SEQUENCES_DICT'] = FEATURENAME+"_evid_string_seq_dict"

        # save param_dict
        with open(FEATURES_DIR + FEATURENAME+"_param_dict.pkl", 'wb') as f:
            pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)

        # save the reconstructed strings
        with open(FEATURES_DIR + param_dict['CLAIM_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'wb') as f:
            pickle.dump(claim_text_full, f, pickle.HIGHEST_PROTOCOL)
        with open(FEATURES_DIR + param_dict['ORIG_DOC_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'wb') as f:
            pickle.dump(orig_doc_text_full, f, pickle.HIGHEST_PROTOCOL)
        with open(FEATURES_DIR + param_dict['EVID_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'wb') as f:
            pickle.dump(evid_text_full, f, pickle.HIGHEST_PROTOCOL)

    # if hiararchical BiLSTM and the Unknown tokens should be in, then the vocab has to be used to reconstruct the sentences
    if "LSTM_100d_1_20_20_zero_emb" in FEATURENAME and with_unknown == True:
        with open(FEATURES_DIR + FEATURENAME+"_param_dict.pkl", "rb") as f:
            param_dict = pickle.load(f)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['CLAIM_VOCAB_FILE'], 'rb') as f:
            CLAIM_VOCAB_FILE = pickle.load(f)
        X_test_claims = np.load(FEATURES_DIR + FEATURENAME + "_claims." + str(fold) + ".npy")
        claim_text_full = misc.sequences_to_text_h_bilstm(X_test_claims, CLAIM_VOCAB_FILE)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['ORIG_DOC_VOCAB_FILE'], 'rb') as f:
            ORIG_DOC_VOCAB_FILE = pickle.load(f)
        X_test_orig_docs = np.load(FEATURES_DIR + FEATURENAME + "_docs." + str(fold) + ".npy")
        orig_doc_text_full = misc.sequences_to_text_h_bilstm_docs(X_test_orig_docs, ORIG_DOC_VOCAB_FILE)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['EVIDENCE_VOCAB_FILE'], 'rb') as f:
            EVIDENCE_VOCAB_FILE = pickle.load(f)
        X_test_evids = np.load(FEATURES_DIR + FEATURENAME + "_evids." + str(fold) + ".npy")
        evid_text_full = misc.sequences_to_text_h_bilstm_docs(X_test_evids, EVIDENCE_VOCAB_FILE)


    if save_file == True:
        # if h_bilstm data with UNKNOWN token is loaded, then do not load the data without UNKNOWN tokens
        if with_unknown == False:
            with open(FEATURES_DIR + FEATURENAME+"_param_dict.pkl", "rb") as f:
                param_dict = pickle.load(f)
            with open(FEATURES_DIR + param_dict['ORIG_DOC_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
                orig_doc_text_full = pickle.load(f)
            with open(FEATURES_DIR + param_dict['CLAIM_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
                claim_text_full = pickle.load(f)
            with open(FEATURES_DIR + param_dict['EVID_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
                evid_text_full = pickle.load(f)


        with open(PATH+"false_positive_samples."+str(fold)+".csv", "w+") as csvfile:
            csvfile.write("Claim,Document,Evidence,Verdict\n")
            for index in list(false_positive_intersection):
                text = "\"%s\",\"%s\",\"%s\"\n" % (str(claim_text_full[index]).replace('"', '\''), str(orig_doc_text_full[index]).replace('"', '\''),
                                                str(evid_text_full[index]).replace('"', '\''))
                csvfile.write(text)

        with open(PATH+"false_negative_samples."+str(fold)+".csv", "w+") as csvfile:
            csvfile.write("Claim,Document,Evidence,Verdict\n")
            for index in list(false_negative_intersection):
                text = "\"%s\",\"%s\",\"%s\"\n" % (str(claim_text_full[index]).replace('"', '\''), str(orig_doc_text_full[index]).replace('"', '\''),
                                                str(evid_text_full[index]).replace('"', '\''))
                csvfile.write(text)

    return false_positive_intersection, false_negative_intersection
    """


def get_prediction_info(predicted_one_hot, predicted_int, y_test, PLOTS_DIR, filename = "test_file"):
    """
    Saves useful information for error analysis in plots directory
    :param predicted_one_hot:
    :param predicted_int:
    :param y_test:
    :param PLOTS_DIR:
    :return:
    """
    def get_info_for_label(label):
        false_dict = {}
        number = 0
        if label == False:
            number = 1
        for i in range(len(predicted_one_hot)):
            false_dict[i] = predicted_one_hot[i][number]
        temp_dict = false_dict
        sorted_index = sorted(false_dict, key=false_dict.get, reverse=True)
        file = str(label) + "\n"
        file += "Index;probability;correct?\n"
        for i in range(len(sorted_index)):
            correct = "No"
            index = sorted_index[i]
            if predicted_int[index] == y_test[index]:
                correct = "Yes"
            file += str(index) + ";" + str(temp_dict[index]) + ";" + correct + "\n"
        print(sorted_index[:5])
        return file, sorted_index

    file = "Predictions True;Predictions False;Correctly predicted?\n"
    max_true_value = 0.0
    max_false_value = 0.0
    max_true_index = -1
    worst_true_index = -1
    max_false_index = -1
    worst_false_index = -1
    for i, pred in enumerate(predicted_one_hot):
        correctly_pred = -1
        if predicted_int[i] == y_test[i]:
            correctly_pred = "Yes"
        else:
            correctly_pred = "No"

        file += str(pred[0]) + ";" + str(pred[1]) + ";" + str(correctly_pred) + "\n"
        if pred[0] > max_true_value:
            max_true_value = pred[0]
            max_true_index = i
            if predicted_int[i] != y_test[i]:
                worst_true_index = i
        if pred[1] > max_false_value:
            max_false_value = pred[1]
            max_false_index = i
            if predicted_int[i] != y_test[i]:
                worst_false_index = i
    file += "\nStatistics\n"
    file += "max_true_value: " + str(max_true_value) + "\n"
    file += "max_true_index: " + str(max_true_index) + "\n"
    file += "max_false_value: " + str(max_false_value) + "\n"
    file += "max_false_index: " + str(max_false_index) + "\n"
    file += "worst_true_index: " + str(worst_true_index) + "\n"
    file += "worst_false_index: " + str(worst_false_index) + "\n"
    file += "===================================================\n"
    file += "===================================================\n"

    info_false, sorted_false = get_info_for_label(False)
    info_true, sorted_true = get_info_for_label(True)
    with open(PLOTS_DIR + filename+".txt", "w+") as text_file:
        text_file.write(file + info_false + info_true)
    return sorted_true, sorted_false, worst_true_index, worst_false_index


def calculate_misclassified_claim_statistics(MODELNAME, fold='holdout', feature='LSTM_100d_1_20_20_zero_emb', with_unknown = True):
    import csv
    from nltk.corpus import stopwords
    from sklearn import feature_extraction
    import nltk
    import pickle
    import string
    from collections import defaultdict
    from fnc.refs.feature_engineering_helper.misc import get_pos_tag_counts
    from fnc.refs.feature_engineering_helper.readability_indices import flesch_kincaid_grade_level, \
        automated_readability_index, mcalpine_eflaw_index, strain_index, flesch_kincaid_reading_ease, coleman_liau_index, gunning_fog_index, \
        lix_index, rix_index, flesch, flesch_kincaid, mtld, hdd
    PATH = INCORRECT_PRED_PATH + MODELNAME + "/"
    final_printout = ""

    # get stopwords list
    stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
    stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
    all_stop_words = stop_words_sklearn.union(stop_words_nltk)

    def get_info(headline, data_dict, result_dict):
        result = headline + ":\n"

        for key, value in data_dict.items():
            if key == 'Claim':
                claims = data_dict[key]
                result += "Result for " + key + "\n"
                avg_UNKNOWN = 0
                word_len_sum = 0
                word_counter_no_punct = 0
                pct_counter = 0
                avg_wlen = 0
                avg_eflaw = 0
                avg_rix = 0
                avg_flesch = 0
                avg_ttr = 0
                avg_NN_count = 0
                avg_PRP_count = 0
                avg_VBD_count = 0
                avg_RB_count = 0
                avg_CD_count = 0
                avg_stop_words_counter = 0
                for claim in claims:
                    avg_UNKNOWN += claim.count('UNKNOWN')
                    tokens = nltk.word_tokenize(claim)

                    # avg_wlen
                    for token in tokens:
                        if token.lower() in all_stop_words:
                            avg_stop_words_counter += 1

                        if token not in string.punctuation:

                            word_len_sum += len(token)
                            word_counter_no_punct += 1
                        else:
                            pct_counter += 1
                    avg_eflaw += mcalpine_eflaw_index(claim)#, word_counter_no_punct)
                    avg_rix += rix_index(claim)
                    avg_flesch += flesch_kincaid(claim)

                    #get some pos tag counts
                    if len(tokens) > 0:
                        NN_count, PRP_count, RB_count, CD_count, VBD_count = get_pos_tag_counts([tokens])
                        avg_NN_count += NN_count
                        avg_PRP_count += PRP_count
                        avg_RB_count += RB_count
                        avg_CD_count += CD_count
                        avg_VBD_count += VBD_count

                # calculate average pos tags
                avg_NN_count = float(avg_NN_count/len(claims))
                avg_PRP_count = float(avg_PRP_count/len(claims))
                avg_RB_count = float(avg_RB_count/len(claims))
                avg_CD_count = float(avg_CD_count/len(claims))
                avg_VBD_count = float(avg_VBD_count/len(claims))

                avg_stop_words_counter = float(avg_stop_words_counter/len(claims))

                if word_counter_no_punct > 0:
                    avg_wlen = float(word_len_sum / word_counter_no_punct)
                avg_pct_counter = float(pct_counter/len(claims))
                avg_claim_tokens = float(word_counter_no_punct / len(claims))
                avg_UNKNOWN = float(avg_UNKNOWN / len(claims))
                avg_eflaw = float(avg_eflaw/len(claims))
                avg_rix = float(avg_rix/len(claims))
                avg_flesch = float(avg_flesch/len(claims))

                result_dict['claim_word'].append(avg_claim_tokens)
                result_dict['claim_UNKNOWN'].append(avg_UNKNOWN)
                result_dict['claim_avg_wlen'].append(avg_wlen)
                result_dict['claim_avg_pct'].append(avg_pct_counter)
                result_dict['claim_avg_eflaw'].append(avg_eflaw)
                result_dict['claim_avg_rix'].append(avg_rix)
                result_dict['claim_avg_flesch'].append(avg_flesch)
                result_dict['claim_avg_ttr'].append(avg_ttr)
                result_dict['claim_avg_NN_count'].append(avg_NN_count)
                result_dict['claim_avg_PRP_count'].append(avg_PRP_count)
                result_dict['claim_avg_RB_count'].append(avg_RB_count)
                result_dict['claim_avg_CD_count'].append(avg_CD_count)
                result_dict['claim_avg_VBD_count'].append(avg_VBD_count)
                result_dict['claim_avg_stop_words_counter'].append(avg_stop_words_counter)
                result += "Avg amount of UNKNOWN words: " + str(avg_UNKNOWN) + "\n"
                result += "Avg word length: " + str(avg_wlen) + "\n"
                result += "Avg number of punctuations: " + str(avg_pct_counter) + "\n"
                result += "Avg mcalpine_eflaw_index: " + str(avg_eflaw) + "\n"
                result += "Avg rix_index: " + str(avg_rix) + "\n"
                result += "Avg flesch: " + str(avg_flesch) + "\n"
                result += "Avg NN_count: " + str(avg_NN_count) + "\n"
                result += "Avg PRP_count: " + str(avg_PRP_count) + "\n"
                result += "Avg RB_count: " + str(avg_RB_count) + "\n"
                result += "Avg CD_count: " + str(avg_CD_count) + "\n"
                result += "Avg VBD_count: " + str(avg_VBD_count) + "\n"
                result += "Avg stop_words_counter: " + str(avg_stop_words_counter) + "\n"
                result += "Avg amount of words: " + str(avg_claim_tokens) + "\n\n"
            else:
                documents = data_dict[key]
                empty_docs = 0
                total_sents = 0
                avg_UNKNOWN = 0
                word_len_sum = 0
                word_counter_no_punct = 0
                pct_counter = 0
                avg_wlen = 0
                avg_rix = 0
                avg_eflaw = 0
                avg_rix_sent = 0
                avg_fk = 0
                avg_mtld = 0
                avg_NN_count = 0
                avg_PRP_count = 0
                avg_VBD_count = 0
                avg_RB_count = 0
                avg_CD_count = 0
                avg_stop_words_counter = 0
                count_mltd_pre_docs = 0 # counts only docs that have at least 50 tokens
                for doc in documents:
                    if len(doc) < 1:
                        empty_docs += 1

                    sents = nltk.sent_tokenize(doc)
                    total_sents += len(sents)
                    token_list = []
                    sent_token_list = []
                    for sent in sents:

                        avg_rix_sent += rix_index(sent)

                        avg_UNKNOWN += sent.count('UNKNOWN')
                        tokens = nltk.word_tokenize(sent)
                        token_list.extend(tokens)
                        sent_token_list.append(tokens)
                        # avg_wlen
                        for token in tokens:
                            if token.lower() in all_stop_words:
                                avg_stop_words_counter += 1

                            if token not in string.punctuation:
                                word_len_sum += len(token)
                                word_counter_no_punct += 1
                            else:
                                pct_counter += 1

                    # claculate mtld
                    if len(token_list) >= 50:
                        count_mltd_pre_docs += 1
                        avg_mtld += mtld(token_list)

                    #get some pos tag counts
                    if len(sent_token_list) > 0:
                        NN_count, PRP_count, RB_count, CD_count, VBD_count = get_pos_tag_counts(sent_token_list)
                        avg_NN_count += NN_count
                        avg_PRP_count += PRP_count
                        avg_RB_count += RB_count
                        avg_CD_count += CD_count
                        avg_VBD_count += VBD_count

                    # calculate readability index
                    avg_rix += rix_index(doc)#, word_counter_no_punct)  # , word_counter_no_punct) # strain_index!
                    avg_eflaw += mcalpine_eflaw_index(doc)#, word_counter_no_punct)  # , word_counter_no_punct) # strain_index!
                    avg_fk += flesch_kincaid(doc)#, word_counter_no_punct)  # , word_counter_no_punct) # strain_index!

                # calculate average word length and average punctuations per sentence
                if word_counter_no_punct > 0:
                    avg_wlen = float(word_len_sum / word_counter_no_punct)
                avg_pct_counter = float(pct_counter/total_sents)

                # calculate avg mtld
                avg_mtld = float (avg_mtld / count_mltd_pre_docs)

                # calculate average pos tags
                avg_NN_count = float(avg_NN_count/(len(documents)-empty_docs))
                avg_PRP_count = float(avg_PRP_count/(len(documents)-empty_docs))
                avg_RB_count = float(avg_RB_count/(len(documents)-empty_docs))
                avg_CD_count = float(avg_CD_count/(len(documents)-empty_docs))
                avg_VBD_count = float(avg_VBD_count/(len(documents)-empty_docs))

                avg_stop_words_counter = float(avg_stop_words_counter/(len(documents)-empty_docs))

                # average readability index
                avg_rix = float(avg_rix/(len(documents)-empty_docs))
                avg_eflaw = float(avg_eflaw/(len(documents)-empty_docs))
                avg_fk = float(avg_fk/(len(documents)-empty_docs))
                avg_rix_sent = float(avg_rix_sent/total_sents)

                result_dict[key+'_sentences'].append(total_sents / (len(documents)-empty_docs))
                result_dict[key+'_words'].append(word_counter_no_punct / (len(documents)-empty_docs))
                result_dict[key+'_word_sent'].append(word_counter_no_punct / total_sents)
                result_dict[key+'_UNKOWN'].append(avg_UNKNOWN / (len(documents)-empty_docs))
                result_dict[key+'_UNKOWN_sent'].append(avg_UNKNOWN / total_sents)
                result_dict[key+'_empty'].append((empty_docs/len(documents))*100)
                result_dict[key+'_avg_wlen'].append(avg_wlen)
                result_dict[key+'_avg_pct'].append(avg_pct_counter)
                result_dict[key+'_avg_ri'].append(avg_rix)
                result_dict[key+'_avg_eflaw'].append(avg_eflaw)
                result_dict[key+'_avg_fk'].append(avg_fk)
                result_dict[key+'_avg_ri_sent'].append(avg_rix_sent)
                result_dict[key+'_avg_mtld'].append(avg_mtld)
                result_dict[key+'_avg_NN_count'].append(avg_NN_count)
                result_dict[key+'_avg_PRP_count'].append(avg_PRP_count)
                result_dict[key+'_avg_RB_count'].append(avg_RB_count)
                result_dict[key+'_avg_CD_count'].append(avg_CD_count)
                result_dict[key+'_avg_VBD_count'].append(avg_VBD_count)
                result_dict[key+'_avg_stop_words_counter'].append(avg_stop_words_counter)
                result += "Result for " + key + "\n"
                result += "Avg amount of sentences: " + str(total_sents / (len(documents)-empty_docs)) + "\n"
                result += "Avg amount of words: " + str(word_counter_no_punct / (len(documents)-empty_docs)) + "\n"
                result += "Avg NN_count: " + str(avg_NN_count) + "\n"
                result += "Avg PRP_count: " + str(avg_PRP_count) + "\n"
                result += "Avg RB_count: " + str(avg_RB_count) + "\n"
                result += "Avg CD_count: " + str(avg_CD_count) + "\n"
                result += "Avg VBD_count: " + str(avg_VBD_count) + "\n"
                result += "Avg stop_words_counter: " + str(avg_stop_words_counter) + "\n"
                result += "Avg avg_mtld: " + str(avg_mtld) + "\n"
                result += "Avg FK Grade: " + str(avg_fk) + "\n"
                result += "Avg EFLAW: " + str(avg_eflaw) + "\n"
                result += "Avg RIX: " + str(avg_rix) + "\n"
                result += "Avg RIX per sent: " + str(avg_rix_sent) + "\n"
                result += "Avg amount of words per sentence: " + str(word_counter_no_punct / total_sents) + "\n"
                result += "Avg amount of UNKNOWN words: " + str(avg_UNKNOWN / (len(documents)-empty_docs)) + "\n"
                result += "Avg amount of UNKNOWN words per sentence: " + str(avg_UNKNOWN / total_sents) + "\n"
                result += "Avg word length: " + str(avg_wlen) + "\n"
                result += "Avg number of punctuation per sentence: " + str(avg_pct_counter) + "\n"
                result += "Empty docs: " + str((empty_docs/len(documents))*100) + "%\n\n"
        return result, result_dict

    def print_results_for(file, headline, result_dict):
        with open(PATH + file, 'r') as csvfile:
            data_dict = defaultdict(list)
            reader = csv.reader(csvfile, delimiter=',')

            headers = next(reader, None)

            for row in reader:
                data_dict['Claim'].append(row[0])
                data_dict['Document'].append(row[1])
                data_dict['Evidence'].append(row[2])

            return get_info(headline, data_dict, result_dict)

    #final_printout += print_results_for("samples_combined_for_stats.csv", "COMBINED MISCLASSIFIED SAMPLES")
    result_dict = defaultdict(list)
    data_string, result_dict = print_results_for("false_negative_samples.holdout.csv", "FALSE NEGATIVE SAMPLES", result_dict)
    final_printout += data_string

    data_string, result_dict = print_results_for("false_positive_samples.holdout.csv", "FALSE POSITIVE SAMPLES", result_dict)
    final_printout += data_string

    # average of false negative and false positive samples
    combined_string = "COMBINED MISCLASSIFIED SAMPLES\n"
    combined_string += "Result for Claim\n"
    combined_string += "Avg word length: " + str((result_dict["claim_avg_wlen"][0]+result_dict["claim_avg_wlen"][1])/2) +"\n"
    combined_string += "Avg mcalpine_eflaw_index: " + str((result_dict["claim_avg_eflaw"][0]+result_dict["claim_avg_eflaw"][1])/2) +"\n"
    combined_string += "Avg rix_index: " + str((result_dict["claim_avg_rix"][0]+result_dict["claim_avg_rix"][1])/2) +"\n"
    combined_string += "Avg avg_flesch: " + str((result_dict["claim_avg_flesch"][0]+result_dict["claim_avg_flesch"][1])/2) +"\n"
    combined_string += "Avg number of punctuation per claim: " + str((result_dict["claim_avg_pct"][0]+result_dict["claim_avg_pct"][1])/2) +"\n"
    combined_string += "Avg amount of UNKNOWN words: " + str((result_dict["claim_UNKNOWN"][0]+result_dict["claim_UNKNOWN"][1])/2) +"\n"
    combined_string += "Avg NN_count: " + str((result_dict["claim_avg_NN_count"][0]+result_dict["claim_avg_NN_count"][1])/2) +"\n"
    combined_string += "Avg PRP_count: " + str((result_dict["claim_avg_PRP_count"][0]+result_dict["claim_avg_PRP_count"][1])/2) +"\n"
    combined_string += "Avg RB_count: " + str((result_dict["claim_avg_RB_count"][0]+result_dict["claim_avg_RB_count"][1])/2) +"\n"
    combined_string += "Avg CD_count: " + str((result_dict["claim_avg_CD_count"][0]+result_dict["claim_avg_CD_count"][1])/2) +"\n"
    combined_string += "Avg VBD_count: " + str((result_dict["claim_avg_VBD_count"][0]+result_dict["claim_avg_VBD_count"][1])/2) +"\n"
    combined_string += "Avg stop_words_counter: " + str((result_dict["claim_avg_stop_words_counter"][0]+result_dict["claim_avg_stop_words_counter"][1])/2) +"\n"
    combined_string += "Avg amount of words: " + str((result_dict["claim_word"][0]+result_dict["claim_word"][1])/2) +"\n\n"
    combined_string += "Result for Document\n"
    combined_string += "Avg amount of sentences: " + str((result_dict["Document_sentences"][0]+result_dict["Document_sentences"][1])/2) +"\n"
    combined_string += "Avg NN_count: " + str((result_dict["Document_avg_NN_count"][0]+result_dict["Document_avg_NN_count"][1])/2) +"\n"
    combined_string += "Avg PRP_count: " + str((result_dict["Document_avg_PRP_count"][0]+result_dict["Document_avg_PRP_count"][1])/2) +"\n"
    combined_string += "Avg RB_count: " + str((result_dict["Document_avg_RB_count"][0]+result_dict["Document_avg_RB_count"][1])/2) +"\n"
    combined_string += "Avg CD_count: " + str((result_dict["Document_avg_CD_count"][0]+result_dict["Document_avg_CD_count"][1])/2) +"\n"
    combined_string += "Avg VBD_count: " + str((result_dict["Document_avg_VBD_count"][0]+result_dict["Document_avg_VBD_count"][1])/2) +"\n"
    combined_string += "Avg stop_words_counter: " + str((result_dict["Document_avg_stop_words_counter"][0]+result_dict["Document_avg_stop_words_counter"][1])/2) +"\n"
    combined_string += "Avg MTLD: " + str((result_dict["Document_avg_mtld"][0]+result_dict["Document_avg_mtld"][1])/2) +"\n"
    combined_string += "Avg EFLAW: " + str((result_dict["Document_avg_eflaw"][0]+result_dict["Document_avg_eflaw"][1])/2) +"\n"
    combined_string += "Avg FK grade: " + str((result_dict["Document_avg_fk"][0]+result_dict["Document_avg_fk"][1])/2) +"\n"
    combined_string += "Avg rix_index: " + str((result_dict["Document_avg_ri"][0]+result_dict["Document_avg_ri"][1])/2) +"\n"
    combined_string += "Avg rix_index per sent: " + str((result_dict["Document_avg_ri_sent"][0]+result_dict["Document_avg_ri_sent"][1])/2) +"\n"
    combined_string += "Avg amount of words: " + str((result_dict["Document_words"][0]+result_dict["Document_words"][1])/2) +"\n"
    combined_string += "Avg amount of words per sentence: " + str((result_dict["Document_word_sent"][0]+result_dict["Document_word_sent"][1])/2) +"\n"
    combined_string += "Avg word length: " + str((result_dict["Document_avg_wlen"][0]+result_dict["Document_avg_wlen"][1])/2) +"\n"
    combined_string += "Avg number of punctuation per sentence: " + str((result_dict["Document_avg_pct"][0]+result_dict["Document_avg_pct"][1])/2) +"\n"
    combined_string += "Avg amount of UNKNOWN words: " + str((result_dict["Document_UNKOWN"][0]+result_dict["Document_UNKOWN"][1])/2) +"\n"
    combined_string += "Avg amount of UNKNOWN words per sentence: " + str((result_dict["Document_UNKOWN_sent"][0]+result_dict["Document_UNKOWN_sent"][1])/2) +"\n"
    combined_string += "Empty docs: " + str((result_dict["Document_empty"][0]+result_dict["Document_empty"][1])/2) +"\n\n"
    combined_string += "Result for Evidence\n"
    combined_string += "Avg amount of sentences: " + str((result_dict["Evidence_sentences"][0]+result_dict["Evidence_sentences"][1])/2) +"\n"
    combined_string += "Avg NN_count: " + str((result_dict["Evidence_avg_NN_count"][0]+result_dict["Evidence_avg_NN_count"][1])/2) +"\n"
    combined_string += "Avg PRP_count: " + str((result_dict["Evidence_avg_PRP_count"][0]+result_dict["Evidence_avg_PRP_count"][1])/2) +"\n"
    combined_string += "Avg RB_count: " + str((result_dict["Evidence_avg_RB_count"][0]+result_dict["Evidence_avg_RB_count"][1])/2) +"\n"
    combined_string += "Avg CD_count: " + str((result_dict["Evidence_avg_CD_count"][0]+result_dict["Evidence_avg_CD_count"][1])/2) +"\n"
    combined_string += "Avg VBD_count: " + str((result_dict["Evidence_avg_VBD_count"][0]+result_dict["Evidence_avg_VBD_count"][1])/2) +"\n"
    combined_string += "Avg stop_words_counter: " + str((result_dict["Evidence_avg_stop_words_counter"][0]+result_dict["Evidence_avg_stop_words_counter"][1])/2) +"\n"
    combined_string += "Avg MTLD: " + str((result_dict["Evidence_avg_mtld"][0]+result_dict["Evidence_avg_mtld"][1])/2) +"\n"
    combined_string += "Avg EFLAW: " + str((result_dict["Evidence_avg_eflaw"][0]+result_dict["Evidence_avg_eflaw"][1])/2) +"\n"
    combined_string += "Avg FK grade: " + str((result_dict["Evidence_avg_fk"][0]+result_dict["Evidence_avg_fk"][1])/2) +"\n"
    combined_string += "Avg rix_index: " + str((result_dict["Evidence_avg_ri"][0]+result_dict["Evidence_avg_ri"][1])/2) +"\n"
    combined_string += "Avg rix_index per sent: " + str((result_dict["Evidence_avg_ri_sent"][0]+result_dict["Evidence_avg_ri_sent"][1])/2) +"\n"
    combined_string += "Avg amount of words: " + str((result_dict["Evidence_words"][0]+result_dict["Evidence_words"][1])/2) +"\n"
    combined_string += "Avg word length: " + str((result_dict["Evidence_avg_wlen"][0]+result_dict["Evidence_avg_wlen"][1])/2) +"\n"
    combined_string += "Avg number of punctuation per sentence: " + str((result_dict["Evidence_avg_pct"][0]+result_dict["Evidence_avg_pct"][1])/2) +"\n"
    combined_string += "Avg amount of words per sentence: " + str((result_dict["Evidence_word_sent"][0]+result_dict["Evidence_word_sent"][1])/2) +"\n"
    combined_string += "Avg amount of UNKNOWN words: " + str((result_dict["Evidence_UNKOWN"][0]+result_dict["Evidence_UNKOWN"][1])/2) +"\n"
    combined_string += "Avg amount of UNKNOWN words per sentence: " + str((result_dict["Evidence_UNKOWN_sent"][0]+result_dict["Evidence_UNKOWN_sent"][1])/2) +"\n"
    combined_string += "Empty docs: " + str((result_dict["Evidence_empty"][0]+result_dict["Evidence_empty"][1])/2) +"\n\n"

    final_printout += combined_string


    false_positive_intersection, false_negative_intersection = get_repeatedly_misqualified_samples(str(fold),
                                                                                                   MODELNAME,
                                                                                                   feature,
                                                                                                   save_file=False, with_unknown = with_unknown)
    misclassified_samples = false_positive_intersection.union(false_negative_intersection)

    if "LSTM_100d_1_20_20_zero_emb" in feature and with_unknown == True:
        with open(FEATURES_DIR + feature+"_param_dict.pkl", "rb") as f:
            param_dict = pickle.load(f)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['CLAIM_VOCAB_FILE'], 'rb') as f:
            CLAIM_VOCAB_FILE = pickle.load(f)
        X_test_claims = np.load(FEATURES_DIR + feature + "_claims." + str(fold) + ".npy")
        claim_text_full = misc.sequences_to_text_h_bilstm(X_test_claims, CLAIM_VOCAB_FILE)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['ORIG_DOC_VOCAB_FILE'], 'rb') as f:
            ORIG_DOC_VOCAB_FILE = pickle.load(f)
        X_test_orig_docs = np.load(FEATURES_DIR + feature + "_docs." + str(fold) + ".npy")
        orig_doc_text_full = misc.sequences_to_text_h_bilstm_docs(X_test_orig_docs, ORIG_DOC_VOCAB_FILE)

        # reconstruct claim string
        with open(FEATURES_DIR + param_dict['EVIDENCE_VOCAB_FILE'], 'rb') as f:
            EVIDENCE_VOCAB_FILE = pickle.load(f)
        X_test_evids = np.load(FEATURES_DIR + feature + "_evids." + str(fold) + ".npy")
        evid_text_full = misc.sequences_to_text_h_bilstm_docs(X_test_evids, EVIDENCE_VOCAB_FILE)

    else:
        with open(FEATURES_DIR + feature + "_param_dict.pkl", "rb") as f:
            param_dict = pickle.load(f)

        with open(FEATURES_DIR + param_dict['ORIG_DOC_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
            orig_doc_text_full = pickle.load(f)
        with open(FEATURES_DIR + param_dict['CLAIM_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
            claim_text_full = pickle.load(f)
        with open(FEATURES_DIR + param_dict['EVID_STRING_SEQUENCES_DICT'] + "." + str(fold) + ".pkl", 'rb') as f:
            evid_text_full = pickle.load(f)

    """for i, claim in claim_text_full.items():
        if 'lumberton' in claim or 'members of the' in claim or 'women should' in claim:
            print(str(i) +": "+claim)"""

    all_indices = set(range(946))

    correctly_classified_indices = all_indices.difference(misclassified_samples)
    data_dict = defaultdict(list)

    # get the text of all correctly classified samples
    for i in correctly_classified_indices:
        data_dict['Claim'].append(claim_text_full[i])
        data_dict['Document'].append(orig_doc_text_full[i])
        data_dict['Evidence'].append(evid_text_full[i])

    string_result, _ = get_info("CORRECTLY CLASSIFIED", data_dict, defaultdict(list))
    final_printout += string_result

    with open(PATH + "comparison." + str(fold) + ".csv", "w+") as csvfile:
        csvfile.write(final_printout)



from keras.callbacks import Callback
import tensorflow as tf
import os
#from tensorflow.contrib.tensorboard.plugins import projector
class TensorBoard(Callback):
    # CHANGES: Comment out sample_weights usage
    """Tensorboard basic visualizations.

    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets# +
                           #self.model.sample_weights
                            )

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

def calculate_class_weight(y_train, no_classes=2):
    # https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
    from sklearn.utils import class_weight

    class_weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {}
    for i in range(no_classes):
        class_weights[i] = class_weight_list[i]
    print(class_weights)
    return class_weights


from keras.callbacks import Callback
import numpy as np
from fnc.refs.utils.score import LABELS, score_submission
class EarlyStoppingOnF1(Callback):
    """
    Prints some metrics after each epoch in order to observe overfitting
                https://github.com/fchollet/keras/issues/5794
                custom metrics: https://github.com/fchollet/keras/issues/2607
    """

    def __init__(self, epochs,
                 X_test_claims,
                 X_test_orig_docs,
                 y_test, loss_filename, epsilon=0.0, min_epoch = 15, X_test_nt=None):
        self.epochs = epochs
        self.patience = 2
        self.counter = 0
        self.prev_score = 0
        self.epsilon = epsilon
        self.loss_filename = loss_filename
        self.min_epoch = min_epoch
        self.X_test_nt = X_test_nt
        #self.print_train_f1 = print_train_f1

        #self.X_train_claims = X_train_claims
        #self.X_train_orig_docs = X_train_orig_docs
        #self.X_train_evid = X_train_evid
        #self.y_train = y_train

        self.X_test_claims = X_test_claims
        self.X_test_orig_docs = X_test_orig_docs
        self.y_test = y_test
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 < self.epochs:
            from sklearn.metrics import f1_score

            # get prediction and convert into list
            if type(self.X_test_orig_docs).__module__ == np.__name__ and type(self.X_test_nt).__module__ == np.__name__:
                predicted_one_hot = self.model.predict([
                    self.X_test_claims,
                    self.X_test_orig_docs,
                    self.X_test_nt
                ])
            elif type(self.X_test_orig_docs).__module__ == np.__name__:
                predicted_one_hot = self.model.predict([
                    self.X_test_claims,
                    self.X_test_orig_docs,
                ])
            else:
                predicted_one_hot = self.model.predict(self.X_test_claims)
            predict = np.argmax(predicted_one_hot, axis=-1)

            """
            predicted_one_hot_train = self.model.predict([self.X_train_claims, self.X_train_orig_docs, self.X_train_evid])
            predict_train = np.argmax(predicted_one_hot_train, axis=-1)

            
            # f1 for train data
            f1_macro_train = ""
            if self.print_train_f1 == True:
                f1_0_train = f1_score(self.y_train, predict_train, labels=[0], average=None)
                f1_1_train = f1_score(self.y_train, predict_train, labels=[1], average=None)
                f1_macro_train = (f1_0_train[0] + f1_1_train[0]) / 2
                print(" - train_f1_(macro): " + str(f1_macro_train))"""

            predicted = [LABELS[int(a)] for a in predict]
            actual = [LABELS[int(a)] for a in self.y_test]
            # calc FNC score
            fold_score, _ = score_submission(actual, predicted)
            max_fold_score, _ = score_submission(actual, actual)
            fnc_score = fold_score / max_fold_score
            print(" - fnc_score: " + str(fnc_score))

            # f1 for test data
            f1_0 = f1_score(self.y_test, predict, labels=[0], average=None)
            f1_1 = f1_score(self.y_test, predict, labels=[1], average=None)
            f1_2 = f1_score(self.y_test, predict, labels=[2], average=None)
            f1_3 = f1_score(self.y_test, predict, labels=[3], average=None)
            f1_macro = (f1_0[0] + f1_1[0] + f1_2[0] + f1_3[0]) / 4
            print(" - val_f1_(macro): " + str(f1_macro))
            print("\n")

            header = ""
            values = ""
            for key, value in logs.items():
                header = header + key + ";"
                values = values + str(value) + ";"
            if epoch == 0:
                values = "\n" + header + "val_f1_macro;" + "fnc_score;" + "\n" + values + str(f1_macro) + str(fnc_score) + ";"
            else:
                values += str(f1_macro) + ";" + str(fnc_score) + ";"
            append_to_loss_monitor_file(values, self.loss_filename)

            if epoch >= self.min_epoch-1:  # 9
                if f1_macro + self.epsilon <= self.prev_score:
                    self.counter += 1
                else:
                    self.counter = 0
                if self.counter >= 2:
                    self.model.stop_training = True
            #print("Counter at " + str(self.counter))
            self.prev_score = f1_macro
            #print("\n")

def get_avg_sent_length(data):
    """
    Calculates how long a sentence is on average in the data
    :param data:
    :return:
    """
    count_non_zeros = 0
    count_sents = 0
    filled_sents = 0
    for batch in data:
        for sent in batch:
            if np.any(sent):
                filled_sents += 1
            count_sents += 1
            for word in sent:
                if word != 0:
                    count_non_zeros += 1
    return [count_non_zeros/count_sents, filled_sents/len(data)]

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):

    """
    Returns output of the layer with name layer_name. Layer output has to be a list. Second item is returned.
    Sources:
        https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_utils.py
        https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
    :param model: The trained model
    :param model_inputs: Input data for the trained model
    :param print_shape_only: If True prints only the shape of the output matrix, otherwise prints the matrix itself
    :param layer_name: Name of the layer for which the output is returned
    :return: Layer output
    """
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output[1] for layer in model.layers if
               layer.name == layer_name] # or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0)
    else:
        list_inputs = [model_inputs, 0]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def display_activations(activation_maps):
    """
    Visualizes the axtivation output
    Sources:
        https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
    :param activation_maps: Activations (outputs of the layer is going to be visualized)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['backend'] = 'TkAgg'
    mpl.rcParams['interactive'] = True

    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()

