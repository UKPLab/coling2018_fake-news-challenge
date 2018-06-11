import pandas as pd
import numpy as np
import os
from fnc.utils.merge2csvs import merge2csvs
from fnc.refs.utils.score import LABELS, score_submission
import fnc.refs.fnc1.scorer as scorer
from fnc.utils import printout_manager
'''
This script provides methods to hardvote using a singe submission file with columns per prediction
See Script "merge2csvs.py to generate such a file
'''

#Note the order of LABELS defines the prediction priority in case of equality when weights are equal
#LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
COLUMNNAME1 = 'Athene mlp hardvoting'
COLUMNNAME2 = 'UCL / Riedel'
COLUMNNAME3 = 'Talos'
LABELS = ['disagree', 'agree', 'discuss', 'unrelated']
LABELWEIGHTS = [ 1.0, 1.0, 1.0, 1.0]
FILECOLUMNS = [COLUMNNAME1, COLUMNNAME2,COLUMNNAME3]
WEIGHTS = [ 1.0, 1.0, 1.0]

MYPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\fnc_results\\meta-ensemble\\'

def hardvote_test(filename1):
    print("Hardvoting with file: " + filename1)
    sumw = sum(WEIGHTS)
    #print(sumw)

    results_dataframe = pd.read_csv(filename1, usecols=['Headline','Body ID', 'Stance', FILECOLUMNS[0],FILECOLUMNS[1],FILECOLUMNS[2]])
    prediction_list = []

    for index, row in results_dataframe.iterrows():
        rowlabels=[0, 0, 0, 0]
        #print('Headline: ' + row['Headline'])
        for x in FILECOLUMNS:
            #print('Columname: ' + x)
            for i in range(4):
                #print('row[x]: ' + row[x])
                #print('LABELS[i]: ' + LABELS[i])
                if row[x] == LABELS[i]:
                    #print('Column:' + str(FILECOLUMNS.index(x)))
                    #print('Prediction matches label:' + row[x])
                    rowlabels[i] += WEIGHTS[FILECOLUMNS.index(x)] * LABELWEIGHTS[i]

        prediction = LABELS[np.argmax(rowlabels)]
        #print(rowlabels)
        #print(prediction)
        prediction_list.append(prediction)

    prediction_dataframe = pd.DataFrame(prediction_list)
    #print(prediction_dataframe)

    df_output = pd.DataFrame()
    df_output['Headline'] = results_dataframe['Headline']
    df_output['Body ID'] = results_dataframe['Body ID']
    df_output['Stance'] = prediction_dataframe[0]

    dest_filename = filename1[:-4]+"_hardvoted_all_equal_1disa_1agr_1disc_1unr.csv"
    df_output.to_csv(dest_filename, index=False, encoding='utf-8')
    print('Saved results to: ' + dest_filename)
    return dest_filename

def mergefiles(number):
    filename1 = MYPATH + "competition_test_stances.csv"
    filename_dest = MYPATH + "ensemble_submission_consolidated" + number+ ".csv"

    filename2 = MYPATH + "athene_submission"+number+".csv"
    test = merge2csvs(filename1, filename2, filename_dest, COLUMNNAME1).mergeFNC()

    filename1 = MYPATH + "ensemble_submission_consolidated" + number + ".csv"

    filename2 = MYPATH + "riedel_submission"+number+".csv"
    test = merge2csvs(filename1, filename2, filename_dest, COLUMNNAME2).mergeFNC()

    filename2 = MYPATH + "talos_submission"+number+".csv"
    test = merge2csvs(filename1, filename2, filename_dest, COLUMNNAME3).mergeFNC()

    return filename_dest

def scorefile(filename):
    """ Calculate Score """
    # calculate FNC Score
    fnc_result_file = filename
    goldlabels_test_file = MYPATH + "competition_test_stances.csv"

    predicted_set = scorer.load_dataset(fnc_result_file)
    fnc_gold_labels = scorer.load_dataset(goldlabels_test_file)
    test_score, cm, f1_score = scorer.score_submission(fnc_gold_labels, predicted_set)
    null_score, max_score = scorer.score_defaults(fnc_gold_labels)

    fnc_results = "################################################ \n"
    fnc_results += "Scorer-type: Manual Hardvoting \n"
    fnc_results += "Scored: " + fnc_result_file + "\n"
    #fnc_results += "Corpora: " + myConstants.datasetName + "\n"
    #fnc_results += "Model:" + scorer_type + myConstants.model_name + "\n"
    fnc_results += printout_manager.calculate_confusion_matrix(cm)
    fnc_results += scorer.SCORE_REPORT.format(max_score, null_score, test_score) + "\n"
    fnc_results += "\n Relative Score: {:.3f}".format(100/max_score*test_score) + "% \n"
    print(fnc_results)
    #printout_manager.save_file(fnc_results, result_file_folder + "/fnc_results.txt", "a+")

if __name__ == '__main__':
    #filename1 = sys.argv[1]
    merged_file = mergefiles(number="10")
    final_file= hardvote_test(merged_file)
    scorefile(final_file)

