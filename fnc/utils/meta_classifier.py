import pandas as pd
import os
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import logistic
from sklearn import naive_bayes
from sklearn.ensemble import VotingClassifier

#import fnc.utils.estimator_definitions as estimator_definitions
from fnc.refs.utils.score import score_submission
import fnc.refs.fnc1.scorer as scorer
from fnc.utils import printout_manager

'''
Classifier implementing an ensemble method to combine the prediction of other classifiers.
'''

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
ATHENECOLUMNS = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
TALOSCOLUMNS = ['prob_0', 'prob_1', 'prob_2', 'prob_3']

def pipeline():

    """ Load data """
    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '\data\\fnc-1\\fnc_results\\meta-ensemble\\'
    # Train
    train_corpus_version = "_8_2"
    athene_train_file = directory + "athene_train_submission" + train_corpus_version + ".csv"
    riedel_train_file = directory + "riedel_train_submission" + train_corpus_version + ".csv"
    talos_train_file = directory + "talos_train_submission" + train_corpus_version + ".csv"
    goldlabels_train_file = directory + "train_test_stances" + train_corpus_version + ".csv"

    athene_train_file_probs = directory + "athene_train_submission" + train_corpus_version + "_probs.csv"
    talos_train_file_probs= directory + "talos_train_submission" + train_corpus_version + "_probs.csv"
    goldlabels_train_file = directory + "train_test_stances" + train_corpus_version + ".csv"

    athene_train_dataframe = pd.read_csv(athene_train_file, usecols=['Headline','Body ID', 'Stance'])
    athene_train_dataframe_probs = pd.read_csv(athene_train_file_probs, usecols=['Headline','Body ID', 'Agree', 'Disagree', 'Discuss', 'Unrelated'])
    riedel_train_dataframe = pd.read_csv(riedel_train_file, usecols=['Headline','Body ID', 'Stance'])
    talos_train_dataframe = pd.read_csv(talos_train_file, usecols=['Headline','Body ID', 'Stance'])
    talos_train_dataframe_probs = pd.read_csv(talos_train_file_probs, usecols=['Headline','Body ID', 'prob_0', 'prob_1', 'prob_2', 'prob_3'])
    goldlabels_train_dataframe = pd.read_csv(goldlabels_train_file, usecols=['Stance'])

    # Test Data
    runnumber = "1"
    athene_test_file = directory + "athene_submission"+runnumber+".csv"
    riedel_test_file = directory + "riedel_submission"+runnumber+".csv"
    talos_test_file = directory + "talos_submission"+runnumber+".csv"
    goldlabels_test_file = directory + "competition_test_stances.csv"

    athene_test_file_probs = directory + "athene_submission"+runnumber+"_probs.csv"
    talos_test_file_probs = directory + "talos_submission"+runnumber+"_probs.csv"
    goldlabels_test_file = directory + "competition_test_stances.csv"

    athene_test_dataframe = pd.read_csv(athene_test_file, usecols=['Headline','Body ID', 'Stance'])
    athene_test_dataframe_probs = pd.read_csv(athene_test_file_probs, usecols=['Headline','Body ID', 'Agree', 'Disagree', 'Discuss', 'Unrelated'])
    riedel_test_dataframe = pd.read_csv(riedel_test_file, usecols=['Headline','Body ID', 'Stance'])
    talos_test_dataframe = pd.read_csv(talos_test_file, usecols=['Headline','Body ID', 'Stance'])
    talos_test_dataframe_probs = pd.read_csv(talos_test_file_probs, usecols=['Headline','Body ID', 'prob_0', 'prob_1', 'prob_2', 'prob_3'])
    goldlabels_test_dataframe = pd.read_csv(goldlabels_test_file, usecols=['Stance'])
    print("Done loading data")

    """ Calculate feats """
    # Train feats
    athene_train_feats = calculate_feats(athene_train_dataframe)
    riedel_train_feats = calculate_feats(riedel_train_dataframe)
    talos_train_feats = calculate_feats(talos_train_dataframe)
    final_train_feats = np.concatenate([np.concatenate([athene_train_feats, riedel_train_feats],axis=1), talos_train_feats], axis=1)

    # Probability feats for athene and talos
    athene_train_feats_probs = calculate_proba_feats(athene_train_dataframe_probs, ATHENECOLUMNS)
    talos_train_feats_probs = calculate_proba_feats(talos_train_dataframe_probs, TALOSCOLUMNS)

    #final_train_feats = np.concatenate([athene_train_feats_probs, talos_train_feats_probs], axis=1)

    # Test feats
    athene_test_feats = calculate_feats(athene_test_dataframe)
    riedel_test_feats = calculate_feats(riedel_test_dataframe)
    talos_test_feats = calculate_feats(talos_test_dataframe)
    final_test_feats = np.concatenate([np.concatenate([athene_test_feats, riedel_test_feats],axis=1), talos_test_feats], axis=1)

    # Probability feats for athene and talos
    athene_test_feats = calculate_proba_feats(athene_test_dataframe_probs, ATHENECOLUMNS)
    talos_test_feats = calculate_proba_feats(talos_test_dataframe_probs, TALOSCOLUMNS)

    #final_test_feats = np.concatenate([athene_test_feats, talos_test_feats],axis=1)
    print("Done calculating feats")


    """ Define estimator """
    scorer_type = 'voting_hard_bayes_gradboost'
    scorer_type = 'grad_boost'
    scorer_type = 'logistic_regression'
    # clf = estimator_definitions.get_estimator(scorer_type)
    clf = getEstimator(scorer_type)


    X_train = final_train_feats
    Y_train = [int(LABELS.index(a['Stance'])) for index, a in goldlabels_train_dataframe.iterrows()]

    """ Train estimator"""
    clf.fit(X_train, Y_train)
    print("Done fitting the estimator")


    X_test = final_test_feats
    Y_test_gold = [int(LABELS.index(a['Stance'])) for index, a in goldlabels_test_dataframe.iterrows()]

    """ Predict """
    #taken fo  benjamins original implementation for athene fnc
    #predict the labes for fitted classifier with the test data
    predicted_int = clf.predict(X_test)
    print("Done predicting")

    predicted = [LABELS[int(a)] for a in predicted_int]
    print(len(predicted))
    actual = [LABELS[int(a)] for a in Y_test_gold]

    # calculate the FNC-1 score based on the predicted and the actual labels
    fold_score, _ = score_submission(actual, predicted)
    max_fold_score, _ = score_submission(actual, actual)
    score = fold_score / max_fold_score

    """ Save results """
    destfilename = "ensemble_submission"+runnumber+".csv"
    destfile = directory + destfilename
    df_output = pd.DataFrame()
    df_output['Headline'] = athene_test_dataframe['Headline']
    df_output['Body ID'] = athene_test_dataframe['Body ID']
    df_output['Stance'] = predicted
    df_output.to_csv(destfile, index=False, encoding='utf-8')


    """ Calculate Score """
    # calculate FNC Score
    fnc_result_file = destfile

    predicted_set = scorer.load_dataset(fnc_result_file)
    fnc_gold_labels = scorer.load_dataset(goldlabels_test_file)
    test_score, cm, f1_score = scorer.score_submission(fnc_gold_labels, predicted_set)
    null_score, max_score = scorer.score_defaults(fnc_gold_labels)

    fnc_results = "################################################ \n"
    fnc_results += "Scorer-type: " + scorer_type + " \n"
    fnc_results += "Saved in: " + destfilename + "\n"
    #fnc_results += "Corpora: " + myConstants.datasetName + "\n"
    #fnc_results += "Model:" + scorer_type + myConstants.model_name + "\n"
    fnc_results += printout_manager.calculate_confusion_matrix(cm)
    fnc_results += scorer.SCORE_REPORT.format(max_score, null_score, test_score) + "\n"
    fnc_results += "\n Relative Score: {:.3f}".format(100/max_score*test_score) + "% \n"
    print(fnc_results)
    #printout_manager.save_file(fnc_results, result_file_folder + "/fnc_results.txt", "a+")


def calculate_feats(submission_dataframe):
    features = []
    for index, row in submission_dataframe.iterrows():
        rowlabels=[0, 0, 0, 0]
        rowlabels[LABELS.index(row['Stance'])] = 1
        #print(index)
        #print(rowlabels)
        features.append(rowlabels)
    return features


def calculate_proba_feats(submission_dataframe, customColumns):
    features = []
    for index, row in (submission_dataframe.iterrows()):
        rowlabels = [0, 0, 0, 0]
        #print(customColumns[0])
        #print(index)
        rowlabels[0] = row[customColumns[0]]
        rowlabels[1] = row[customColumns[1]]
        rowlabels[2] = row[customColumns[2]]
        rowlabels[3] = row[customColumns[3]]
        features.append(rowlabels)
    return features


#taken from original Athene submission
def getEstimator(scorer_type):
    if scorer_type == 'grad_boost':
        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

    if scorer_type == 'svm1': # stochastic gradient decent classifier
        clf = svm.SVC(gamma=0.001, C=100., verbose=True)

    if scorer_type == 'logistic_regression' :
        clf = logistic.LogisticRegression()

    if scorer_type == 'svm3':
        clf = svm.SVC(kernel='poly', C=1.0, probability=True, class_weight='unbalanced')

    if scorer_type == "bayes":
        clf = naive_bayes.GaussianNB()

    if scorer_type == 'voting_hard_svm_gradboost_logistic':
        svm2 = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', verbose=True)
        log_reg = logistic.LogisticRegression()
        gradboost = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

        clf = VotingClassifier(estimators=[  # ('gb', gb),
            ('svm', svm2),
            ('grad_boost', gradboost),
            ('logisitc_regression', log_reg)
        ],  n_jobs=1,
            voting='hard')

    if scorer_type == 'voting_hard_bayes_gradboost':
        bayes = naive_bayes.GaussianNB()
        gradboost = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

        clf = VotingClassifier(estimators=[  # ('gb', gb),
            ('bayes', bayes),
            ('grad_boost', gradboost),
        ],  n_jobs=1,
            voting='hard')

    return clf

if __name__ == '__main__':
    pipeline()
