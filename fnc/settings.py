import os.path as path
from fnc.refs.utils.dataset import DataSet

HOME = "~/"

## In this class you can define the parameters to be used
# - perform_oversampling (define the oversampling or undersampling method in pipeline.py line 623ff)
# - model_name: After the pipeline has been called with ftrain, a model is saved to disc and the name is shown on the commandline
#               This parameter is used to define which model shall be used when the pipeline is called with ftest.
# - corpus selection: This parameter selects a corpus for ftrain and ftest. The definition of the different corpora can be found below
# - feature_list: This array contains  (multiple lists of)
#       1. the name of the model to be used (models are defined in utils/
#       2. an array for normal features
#       3. an array for non-bleeding features

class myConstants():
    corpusSelection = 0
    '''
    0 = FNC train data fr training, FNC test data for testing
    1 = ARC train data for training, ARC test data for testing
    2 = ARC train data for training, FNC test data for testing
    3 = FNC train data for training, ARC test data for testing
    4 = ARC + FNC train data combined for training, ARC+FNC test data combined for testing
    '''

    perform_oversampling = False
    # define the oversampling or undersampling method in pipeline.py line 623ff

    model_name = "_final_new_11/"
    # note that the full model name is a concatenation of the model selected in the following "feature_list" and the suffix above.
    # each time a model is trained, the number is increased

    #the following feature list contains the "original Athene" features
    feature_list = [
        # 30 'featMLP'
        ('featMLP',
         ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent'],
         [])
    ]


    '''
    feature_list = [
        # LSTM
        ('stackLSTM',
         ['single_flat_LSTM_50d_100', 'overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent'],
         [])
    ]

    feature_list = [
        # 30 'featMLP' + discuss
        ('featMLP',
         ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent',
          'discuss', 'hedging'],
         [])
    ]
    '''

    """
    Groups of features for ablation test. Listed are the lists for normal and for non-bleeding features.
    """
    bow_feats = ['hand', 'negated_context_word_12grams_concat_tf5000_l2_all_data', 'char_3grams_5000_concat_all_data']

    #topic_models = ['latent_dirichlet_allocation_300', 'latent_semantic_indexing_gensim_300_concat', 'NMF_fit_all_concat_300_no_holdout', 'NMF_cos_300']
    topic_models = ['latent_dirichlet_allocation_300', 'latent_semantic_indexing_gensim_300_concat_holdout', 'NMF_fit_all_concat_300_and_test', 'NMF_cos_300']
    lexicon_feats = ['refuting', 'polarity', 'nrc_emo_lex', 'sentiment140_unigrams',
                     'nrc_hashtag_sentiment_unigram', 'mpqa_unigrams', 'max_diff_twitter_uni_bigrams']

    readab_feats = ['readability_features']

    lexical_feats = ['overlap', 'lexical_features']

    #stanford_feats = ['stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent']
    stanford_feats = ['stanford_ppdb', 'stanford_wordsim']

    POS_feats = stanford_feats + ['nrc_hashtag_sentiment_unigram_POS']
    #POS_feats_new = ['stanford_wordsim_1sent']

    structural_feats = ['structural_features']
    '''
    feature_list = [
        ## Topic Models
        ('MLP_base_1',['latent_semantic_indexing_gensim_300_concat_holdout'],[]),
        ('MLP_base_1',['NMF_cos_300'],[]),
        ('MLP_base_1',['NMF_fit_all_concat_300_and_test'],[]),
        ('MLP_base_1',['latent_dirichlet_allocation_300'],[]),

        ## BoW/BoC features
        ('MLP_base_1',['hand'],[]),
        ('MLP_base_1',['negated_context_word_12grams_concat_tf5000_l2_all_data'],[]),
        ('MLP_base_1',['char_3grams_5000_concat_all_data'],[]),

        ## POS Feats alone
        ('MLP_base_1', stanford_feats, []),
        ('MLP_base_1', ['nrc_hashtag_sentiment_unigram_POS'], []),

        ##Lexicon based features
        ('MLP_base_1', ['nrc_emo_lex'], []),
        ('MLP_base_1', ['sentiment140_unigrams'], []),
        ('MLP_base_1', ['nrc_hashtag_sentiment_unigram'], []),
        ('MLP_base_1', ['mpqa_unigrams'], []),
        ('MLP_base_1', ['polarity'], []),
        ('MLP_base_1', ['max_diff_twitter_uni_bigrams'], []),
        ('MLP_base_1', ['refuting'], []),

        ## Readability feats
        ('MLP_base_1', readab_feats, []),

        ## Structural feats
        ('MLP_base_1', structural_feats, []),

        ## Lexical feats
        ('MLP_base_1', lexical_feats, []),

        ## Ablation tests
        # only - tests
        # only bow boc
        ('MLP_base_1', bow_feats, []),
        # only topic
        ('MLP_base_1', topic_models, []),
        # only POS - stanford = pos - hastag
        ('MLP_base_1', stanford_feats, []),

        # all w/o X tests
        # except POS
        ('MLP_base_1', topic_models + bow_feats,
         []),
        # except topic
        ('MLP_base_1', bow_feats + stanford_feats,
         []),
        # except bow boc
        ('MLP_base_1', topic_models + stanford_feats,
         []),

        # All features that are not more than 15% below FNC-1 baseline
        ('MLP_base_1', topic_models+bow_feats+stanford_feats,
         []),

        # all features without pre-selection
        ('MLP_base_1',
         topic_models + bow_feats + lexicon_feats + readab_feats + lexical_feats + POS_feats + structural_feats,
         []),
        ]
    '''
    ## Best score after ablation tests
    feature_list = [
        # all w/o X tests
        # except POS
        ('featMLP', topic_models + bow_feats,
         []),
        ]

    ## Baseline test
    #feature_list = [('grad_boost', ['overlap', 'refuting', 'polarity', 'hand'], [])]

    if corpusSelection == 0:
        # pure FNC
        # train: FNC train; test: FNC testdata
        datasetName = "FNC_FNC"
        splits_dir = "%s/data/fnc-1/splits/FNC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/corpora/FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/corpora/FNC/competition_test_stances.csv"

    elif corpusSelection == 1:
        # pure ARC
        # train: ARC train, test ARC testdata
        datasetName = "ARC_ARC"
        splits_dir = "%s/data/fnc-1/splits/ARC_ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/ARC_ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/corpora/ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "arc_bodies.csv"
        train_stances = "arc_stances_train.csv"
        test_bodies = "arc_bodies.csv"
        test_stances = "arc_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/corpora/ARC/arc_stances_test.csv"

    elif corpusSelection == 2:
        # train: ARC, test: FNC
        datasetName = "ARC_FNC"
        splits_dir = "%s/data/fnc-1/splits/ARC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/ARC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "/corpora/ARC/arc_bodies.csv"
        train_stances = "/corpora/ARC/arc_stances_train.csv"
        test_bodies = "/corpora/FNC/test_bodies.csv"
        test_stances = "/corpora/FNC/test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/corpora/FNC/competition_test_stances.csv"

    elif corpusSelection == 3:
        # train: FNC, test: ARC
        datasetName = "FNC_ARC"
        splits_dir = "%s/data/fnc-1/splits/FNC_ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "/corpora/FNC/train_bodies.csv"
        train_stances = "/corpora/FNC/train_stances.csv"
        test_bodies = "/corpora/ARC/arc_bodies.csv"
        test_stances = "arc_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/corpora/ARC/arc_stances_test.csv"

    elif corpusSelection == 4:
        # combined ARC+FNC
        # train: FNC train+ARCtrain, test FNC test+ ARC test
        datasetName = "FNCARK_FNCARK"
        splits_dir = "%s/data/fnc-1/splits/FNCARC_FNCARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNCARC_FNCARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/corpora/FNC_ARC" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "combined_bodies_train.csv"
        train_stances = "combined_stances_train.csv"
        test_bodies = "combined_bodies_test.csv"
        test_stances = "combined_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/corpora/FNC_ARC/combined_stances_test.csv"

    d = DataSet(data_path, train_bodies, train_stances, datasetName)
    testdataset = DataSet(data_path, test_bodies, test_stances, datasetName)
