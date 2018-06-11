import pandas as pd
import sys, os

'''
This scripts provides methods to combine / merge different csvs
See the comments per ethod for details
'''

class merge2csvs():
    def __init__(self, file1, file2, destfile, newname=""):
        self.file1 = file1
        self.file2 = file2
        self.destfile = destfile
        self.newname = newname


    def append(self):
        '''Use this method to append lines from file2  to file1'''
        myfile1 = pd.read_csv(self.file1, encoding='utf-8')
        myfile2 = pd.read_csv(self.file2, encoding='utf-8')
        merged = pd.concat([myfile1, myfile2], axis=1)
        merged.to_csv(self.destfile, index=False, encoding='utf-8')

    def combine(self):
        '''Use this method to combine stances and bodies to one file to get human upperbound format'''
        myfile1 = pd.read_csv(self.file1, encoding='utf-8')
        myfile2 = pd.read_csv(self.file2, encoding='utf-8')
        merged = myfile1.set_index('Body ID').join(myfile2.set_index('Body ID'), how="left")
        merged.to_csv(self.destfile, encoding='utf-8', index=False,  columns=["Headline", "articleBody", "Stance"])

    def mergeUpperBound(self):
        '''Use this method to combine several predicitons in FNC Human upper boud format (column name "body" instead of "Body ID" '''
        myfile1 = pd.read_csv(self.file1, encoding='utf-8')
        myfile2 = pd.read_csv(self.file2, encoding='utf-8')
        myfile2.rename(columns={"Stance": self.newname}, inplace=True)
        #merged = myfile1.merge(myfile2[["Body", newname]], how="left")
        #merged = myfile1.set_index(['Headline', "Body"]).join(myfile2.set_index(['Headline', "Body"]))
        merged = myfile1.merge(myfile2, on=["Headline", "Body"])
        print(self.destfile + " length: " + str(len(merged)))
        merged.to_csv(self.destfile, encoding='utf-8', index=False)
        #merged.to_csv(self.destfile, encoding='utf-8', index=False,  columns=["Headline", "articleBody", "Stance"])

    def mergeFNC(self):
        '''Use this method to combine several predicitons in FNC format'''
        myfile1 = pd.read_csv(self.file1, encoding='utf-8', warn_bad_lines=True)
        myfile2 = pd.read_csv(self.file2, encoding='utf-8', warn_bad_lines=True)
        myfile2.rename(columns={"Stance": self.newname}, inplace=True)
        merged = pd.concat([myfile1, myfile2[[self.newname]]], axis=1)
        #merged = myfile1.merge(myfile2, on=["Headline", "Body ID"]).drop_duplicates()
        print(self.destfile + " (Destfile) length: " + str(merged.shape[0]))
        merged.to_csv(self.destfile, encoding='utf-8', index=False)

    def createUnlabeled(self):
        myfile1 = pd.read_csv(self.file1, encoding='utf-8', warn_bad_lines=True)
        myfile1.to_csv(self.destfile, encoding='utf-8', index=False, columns=['Headline', 'Body ID'])

if __name__ == '__main__':
    type = sys.argv[1]

    if type == "append":
        filename1 = sys.argv[2]
        filename2 = sys.argv[3]
        filename_dest = sys.argv[4]
        print("Appending filename1: \"" + filename1 + "\" with  filename2: \"" + filename2 + "\"  as: \"" + filename_dest + "\"")
        test = merge2csvs(filename1, filename2, filename_dest).append()
    elif type == "combine":
        filename1 = sys.argv[2]
        filename2 = sys.argv[3]
        filename_dest = sys.argv[4]
        print("Combining filename1: \"" + filename1 + "\" with  filename2: \"" + filename2 + "\"  into: \"" + filename_dest + "\"")
        test = merge2csvs(filename1, filename2, filename_dest).combine()
    elif type == "merge":
        filename1 = sys.argv[2]
        filename2 = sys.argv[3]
        filename_dest = sys.argv[4]
        newname = sys.argv[5]
        print("Merging filename1: \"" + filename1 + "\" with  filename2: \"" + filename2 + "\"  as: \"" + filename_dest + "\"")
        print("New name in csv file: " + newname)
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()
    elif type == "hardcoded_upper_bound":
        mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\upper-bound\\'
        filename1 = mypath + "upper_bound_split.csv"
        filename_dest = mypath + "upper_bound_summary_V2.csv"

        filename2 = mypath + "upper_bound_split_mlp_combined.csv"
        newname = 'Athene mlp'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename1 = mypath + "upper_bound_summary_V2.csv"

        filename2 = mypath + "upper_bound_split_unlabelled_avinesh.csv"
        newname = 'avinesh'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_labelled_andreas.csv"
        newname = 'andreas'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_unlabelled_andreas_v2.csv"
        newname = 'andreas V2'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_labelled_debanjan.csv"
        newname = 'debanjan'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_labelled_debanjan_V2.csv"
        newname = 'debanjan V2'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_unlabelled_benjamin.csv"
        newname = 'Benjamin'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        filename2 = mypath + "upper_bound_split_unlabelled_benjamin_v2.csv"
        newname = 'Benjamin V2'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

        #filename2 = mypath + "upper_bound_split_unlabelled_felix.csv"
        #newname = 'felix'
        #test = merge2csvs(filename1, filename2, filename_dest, newname).merge()

        filename2 = mypath + "upper_bound_split_unlabelled_felix_v2.csv"
        newname = 'felix V2'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeUpperBound()

    elif type == "hardcoded_fnc_test":
        mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\fnc_results\\manual_copied\\'
        filename1 = mypath + "competition_test_stances.csv"
        filename_dest = mypath + "fnc_summary.csv"

        filename2 = mypath + "athene_submission.csv"
        newname = 'Athene mlp hardvoting'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeFNC()

        filename1 = mypath + "fnc_summary.csv"

        filename2 = mypath + "riedel_submission7.csv"
        newname = 'UCL / Riedel'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeFNC()

        filename2 = mypath + "talos_submission12.csv"
        newname = 'Talos'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeFNC()

    elif type == "hardcoded_fnc_test_riedel":
        mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\fnc_results\\meta-ensemble\\'
        mypath2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+'\data\\fnc-1\\metaclassifier_corpus_8_2\\'
        filename1 = mypath2 + "train_test_stances_unlabeled.csv"
        riedel_number = ""
        filename_dest = mypath + "riedel_train_submission_8_2.csv"

        filename2 = mypath + "riedel_unaligned\\riedel_train_stances_8_2.csv"
        newname = 'Stance'
        test = merge2csvs(filename1, filename2, filename_dest, newname).mergeFNC()



    elif type == "createunlabeled":
        filename1 = sys.argv[2]
        filename2 = ""
        filename_dest = sys.argv[3]
        test = merge2csvs(filename1, filename2, filename_dest).createUnlabeled()
    else:
        print("Wrong type given, please select either append or combine")

