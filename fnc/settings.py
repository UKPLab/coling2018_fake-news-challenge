import os.path as path
from fnc.refs.utils.dataset import DataSet

HOME = "~/"


class myConstants():
    perform_oversampling = False
    model_name = "_final_new_11/"
    corpusSelection = 0

    feature_list = [
        # 30 'voting_mlps_hard'
        ('voting_mlps_hard',
         ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent'],
         [])
    ]
    '''
    feature_list = [
        # LSTM
        ('single_f_ext_LSTM_att_no_cw',
         ['single_flat_LSTM_50d_100', 'overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent'],
         [])
    ]

    feature_list = [
        # 30 'voting_mlps_hard' + discuss
        ('voting_mlps_hard',
         ['overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_incl_holdout_and_test',
          'latent_dirichlet_allocation_incl_holdout_and_test', 'latent_semantic_indexing_gensim_holdout_and_test',
          'NMF_fit_all_concat_300_and_test', 'word_ngrams_concat_tf5000_l2_w_holdout_and_test',
          'stanford_ppdb', 'stanford_sentiment_3sent', 'stanford_wordsim' , 'stanford_negation_3sent', 'stanford_avg_words_per_sent_3sent',
          'discuss', 'hedging'],
         [])
    ]

    feature_list = [
        # Benjamins LSTM features
        ('single_f_ext_lstm_att_no_cw',
         ['single_flat_LSTM_50d_100', 'overlap', 'refuting', 'polarity', 'hand', 'NMF_fit_all_concat_300_no_holdout', 'NMF_cos_300',
          'latent_dirichlet_allocation_100',
          'latent_semantic_indexing_gensim_300_concat', 'stanford_wordsim_1sent'],
         ['word_unigrams_5000_concat_tf_l2_no_bleeding'])
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

    ## Best score after ablation tests
    feature_list = [
        # all w/o X tests
        # except POS
        ('voting_mlps_hard', topic_models + bow_feats,
         []),
        ]

    ## Baseline test
    feature_list = [('grad_boost', ['overlap', 'refuting', 'polarity', 'hand'], [])]

    if corpusSelection == 0:
        # train: FNC Corpus; test: FNC testdata
        datasetName = "FNC_FNC"
        splits_dir = "%s/data/fnc-1/splits/FNC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"

    elif corpusSelection == 1:
        # train: UKP corpus; test: FNC testdata
        datasetName = "UKP_FNC"
        splits_dir = "%s/data/fnc-1/splits/UKP_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_bodies.csv"
        train_stances = "ukp_stances.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"

    elif corpusSelection == 2:
        # train: FNC Corpus , test: UKP testdata
        datasetName = "FNC_UKP"
        splits_dir = "%s/data/fnc-1/splits/FNC_UKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_UKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "ukp_bodies.csv"
        test_stances = "ukp_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances.csv"

    elif corpusSelection == 3:
        # train: UKP Corpus , test: UKP testdata
        datasetName = "UKP_UKP"
        splits_dir = "%s/data/fnc-1/splits/UKP_UKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_UKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_bodies.csv"
        train_stances = "ukp_stances_train.csv"
        test_bodies = "ukp_bodies.csv"
        test_stances = "ukp_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances_test.csv"

    elif corpusSelection == 4:
        # train: UKP Corpus , test: UKP testdata
        datasetName = "UKP_small_UKP_small"
        splits_dir = "%s/data/fnc-1/splits/UKP_small_UKP_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_small_UKP_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_bodies_small.csv"
        train_stances = "ukp_stances_small_train.csv"
        test_bodies = "ukp_bodies_small.csv"
        test_stances = "ukp_stances_small_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances_small_test.csv"


    elif corpusSelection == 5:
        # train: UKP Corpus big, test: UKP testdata big
        datasetName = "UKP_big_UKP_big"
        splits_dir = "%s/data/fnc-1/splits/UKP_big_UKP_big" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_big_UKP_big" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_bodies_big.csv"
        train_stances = "ukp_stances_big_train.csv"
        test_bodies = "ukp_bodies_big.csv"
        test_stances = "ukp_stances_big_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances_big_test.csv"

    elif corpusSelection == 6:
        # train: UKP Corpus discuss, test: UKP testdata discuss
        datasetName = "UKP_discuss_UKP_discuss"
        splits_dir = "%s/data/fnc-1/splits/UKP_discuss_UKP_discuss" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_discuss_UKP_discuss" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_bodies_incl_discuss.csv"
        train_stances = "ukp_stances_incl_discuss_train.csv"
        test_bodies = "ukp_bodies_incl_discuss.csv"
        test_stances = "ukp_stances_incl_discuss_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances_incl_discuss_test.csv"

    elif corpusSelection == 7:
        # train: FNC, test: UKP testdata discuss
        datasetName = "FNC_UKP_discuss"
        splits_dir = "%s/data/fnc-1/splits/FNC_UKP_discuss" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_UKP_discuss" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "ukp_bodies_incl_discuss.csv"
        test_stances = "ukp_stances_incl_discuss_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_stances_incl_discuss.csv"

    elif corpusSelection == 8:
        # train: UKP Corpus discuss, test: UKP testdata discuss
        datasetName = "UKP_discuss_small_UKP_discuss_small"
        splits_dir = "%s/data/fnc-1/splits/UKP_discuss_small_UKP_discuss_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_discuss_small_UKP_discuss_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_discuss_small_bodies.csv"
        train_stances = "ukp_discuss_small_stances_train.csv"
        test_bodies = "ukp_discuss_small_bodies.csv"
        test_stances = "ukp_discuss_small_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_discuss_small_stances_test.csv"

    elif corpusSelection == 9:
        # train: FNC, test: UKP testdata discuss
        datasetName = "FNC_UKP_discuss_small"
        splits_dir = "%s/data/fnc-1/splits/FNC_UKP_discuss_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_UKP_discuss_small" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "ukp_discuss_small_bodies.csv"
        test_stances = "ukp_discuss_small_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_discuss_small_stances.csv"

    elif corpusSelection == 10:
        # train: UKP Corpus discuss, test: FNC
        datasetName = "UKP_discuss_small_FNC"
        splits_dir = "%s/data/fnc-1/splits/UKP_discuss_small_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_discuss_small_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_discuss_small_bodies.csv"
        train_stances = "ukp_discuss_small_stances.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"
    #################################################################################
    elif corpusSelection == 11:
        # train: FNC train+UKPtrain, test FNC test+ UKP test
        datasetName = "FNCUKP_FNCUKP"
        splits_dir = "%s/data/fnc-1/splits/FNCUKP_FNCUKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNCUKP_FNCUKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "combined_bodies_train.csv"
        train_stances = "combined_stances_train.csv"
        test_bodies = "combined_bodies_test.csv"
        test_stances = "combined_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/combined_stances_test.csv"

    elif corpusSelection == 12:
        # train: UKPtrain, test UKP test
        datasetName = "UKP_new_UKP_new"
        splits_dir = "%s/data/fnc-1/splits/UKP_new_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_new_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_new_bodies.csv"
        train_stances = "ukp_new_stances_train.csv"
        test_bodies = "ukp_new_bodies.csv"
        test_stances = "ukp_new_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_new_stances_test.csv"

    elif corpusSelection == 13:
        # train: UKPtrain, test UKP test
        # train: FNC train+UKPtrain, test FNC test+ UKP test
        # same as 11 for train but FNC for test
        datasetName = "FNCUKP_FNC"
        splits_dir = "%s/data/fnc-1/splits/FNCUKP_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNCUKP_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "combined_bodies_train.csv"
        train_stances = "combined_stances_train.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"

    elif corpusSelection == 14:
        # same as 12 for train, but FNC for test
        datasetName = "UKP_new_FNC"
        splits_dir = "%s/data/fnc-1/splits/UKP_new_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_new_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_new_bodies.csv"
        train_stances = "ukp_new_stances_train.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"

    elif corpusSelection == 15:
        # train: FNC train+UKPtrain, + UKP test
        datasetName = "FNCUKP_UKP_new"
        splits_dir = "%s/data/fnc-1/splits/FNCUKP_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNCUKP_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "combined_bodies_train.csv"
        train_stances = "combined_stances_train.csv"
        test_bodies = "ukp_new_bodies.csv"
        test_stances = "ukp_new_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_new_stances_test.csv"

    elif corpusSelection == 16:
        # train: UKPtrain, test FNC+UKP test
        datasetName = "UKP_new_FNCUKP"
        splits_dir = "%s/data/fnc-1/splits/UKP_new_FNCUKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/UKP_new_FNCUKP" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "ukp_new_bodies.csv"
        train_stances = "ukp_new_stances_train.csv"
        test_bodies = "combined_bodies_test.csv"
        test_stances = "combined_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/combined_stances_test.csv"


    elif corpusSelection == 17:
        datasetName = "FNC_FNC_human_test"
        splits_dir = "%s/data/fnc-1/splits/FNC_FNC_human_test" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_FNC_human_test" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "upper-bound/aligned/upper_bound_split_unlabelled_aligned_bodies.csv"
        test_stances = "upper-bound/aligned/upper_bound_split_unlabelled_aligned_stances.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/upper-bound/aligned/upper_bound_split_gold_aligned_stances.csv"

    elif corpusSelection == 18:
        # train: FNC Corpus wo unrelated; test: FNC testdata wo unrelated
        datasetName = "FNC_wo_unrelated_FNC_wo_unrelated"
        splits_dir = "%s/data/fnc-1/splits/FNC_wo_unrelated_FNC_wo_unrelated" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_wo_unrelated_FNC_wo_unrelated" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances_wo_unrelated.csv"
        test_bodies = "competition_test_bodies.csv"
        test_stances = "competition_test_stances_wo_unrelated_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances_wo_unrelated.csv"

    elif corpusSelection == 19:
        # train: Meta-Train-corpus; test: Meta-Testcorpus_82
        datasetName = "Meta-Classifier"
        splits_dir = "%s/data/fnc-1/splits/metaclassifier_9_1" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/metaclassifier_9_1" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/metaclassifier_corpus_9_1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_train_stances.csv"
        test_bodies = "train_bodies.csv"
        test_stances = "train_test_stances.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/metaclassifier_corpus_9_1/train_test_stances.csv"

    elif corpusSelection == 20:
        # train: Meta-Train-corpus; test: Meta-Testcorpus_82
        datasetName = "Meta-Classifier"
        splits_dir = "%s/data/fnc-1/splits/metaclassifier_8_2" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/metaclassifier_8_2" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/metaclassifier_corpus_8_2" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_train_stances.csv"
        test_bodies = "train_bodies.csv"
        test_stances = "train_test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/metaclassifier_corpus_8_2/train_test_stances.csv"

    elif corpusSelection == 21:
        # train: Meta-Train-corpus; test: FNC-Testcorpus
        datasetName = "Meta-Classifier"
        splits_dir = "%s/data/fnc-1/splits/metaclassifier_8_2_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/metaclassifier_8_2_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/metaclassifier_corpus_8_2_FNC" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_train_stances.csv"
        test_bodies = "test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/competition_test_stances.csv"

    elif corpusSelection == 22:
        # train: 80%FNC Testcorpus; test: 20%FNC-Testcorpus
        datasetName = "Testset for training"
        splits_dir = "%s/data/fnc-1/splits/testset_for_training" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/testset_for_training" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1/testset_for_training" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "competition_test_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "competition_test_bodies.csv"
        test_stances = "test_stances_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/testset_for_training/test_stances.csv"

    elif corpusSelection == 23:
        # train: FNC , test UKP_new
        datasetName = "FNC_UKP_new"
        splits_dir = "%s/data/fnc-1/splits/FNC_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        features_dir = "%s/data/fnc-1/features/FNC_UKP_new" % (path.dirname(path.dirname(path.abspath(__file__))))
        data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.abspath(__file__))))
        train_bodies = "train_bodies.csv"
        train_stances = "train_stances.csv"
        test_bodies = "ukp_new_bodies.csv"
        test_stances = "ukp_new_stances_test_unlabeled.csv"
        test_stances_gold = path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/fnc/data/fnc-1/ukp_new_stances_test.csv"

    d = DataSet(data_path, train_bodies, train_stances, datasetName)
    testdataset = DataSet(data_path, test_bodies, test_stances, datasetName)
