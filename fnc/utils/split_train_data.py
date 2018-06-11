import os
import pandas as pd
from sklearn.model_selection import train_test_split

'''
This script provides methods to split the train.csv into another train and test set
this new set will be used to train another meta classifier based on the predictions of the FNC classifier.
'''

def split_corpus(sourcefile, destfolder):
    print("reading source-file: " + sourcefile)
    sourcefile_dataframe = pd.read_csv(sourcefile, usecols=['Headline','Body ID', 'Stance'])

    traindata, testdata = train_test_split(sourcefile_dataframe, test_size=0.2)

    traindata_df = pd.DataFrame(traindata)
    testdata_df = pd.DataFrame(testdata)

    print("Destination files are saved to: " + destfolder)
    traindata_df.to_csv(destfolder+"train_train_stances.csv", index=False, encoding='utf-8')
    testdata_df.to_csv(destfolder+"train_test_stances.csv", index=False, encoding='utf-8')
    testdata_df.to_csv(destfolder+"train_test_stances_unlabeled.csv", index=False, encoding='utf-8', columns=['Headline', 'Body ID'])
    print("Done!")

if __name__ == '__main__':
    #sourcefile = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\train_stances.csv'
    #destfolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\metaclassifier_corpus_8_2\\'

    sourcefile = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\competition_test_stances.csv'
    destfolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\testset_for_training\\'

    test = split_corpus(sourcefile, destfolder)

