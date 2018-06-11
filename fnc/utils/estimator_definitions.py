from fnc.models.MultiThreadingFeedForwardMLP import MultiThreadingFeedForwardMLP
#from fnc.models.original_MultiThreadingFeedForwardMLP import MultiThreadingFeedForwardMLP
from fnc.models.riedel_mlp import riedel_mlp
from fnc.models.single_f_ext_LSTM import single_f_ext_LSTM, single_f_ext_LSTM_att
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import logistic
from sklearn import naive_bayes
import numpy as np

def get_estimator(scorer_type, save_folder=None):
    #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

    if scorer_type == 'voting_mlps_hard' or scorer_type == 'featMLP':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        mlp1 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp2 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp3 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp4 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp5 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=seed)


        clf = VotingClassifier(estimators=[  # ('gb', gb),
            # ('mlp', mlp),
            ('mlp1', mlp1),
            ('mlp2', mlp2),
            ('mlp3', mlp3),
            ('mlp4', mlp4),
            ('mlp5', mlp5),
        ],  n_jobs=1,
            voting='hard')



    if scorer_type == 'MLP_base':
        clf = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=200, hm_epochs=30, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.01,
                                           hidden_layers=(600, 600, 600), activation_function='relu', save_folder=save_folder, seed=12345)

    if scorer_type == 'MLP_base_1':
        clf = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                                  learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                                  hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                                  save_folder=save_folder, seed=12345)

    if scorer_type == 'MLP_base_2':
        clf = MultiThreadingFeedForwardMLP(n_classes=3, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                           learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                           hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                           save_folder=save_folder, seed=12345)

    if scorer_type == 'riedel':
        clf = riedel_mlp(save_folder=save_folder)

    #taken from original implementation
    if scorer_type == 'svm2':
        clf = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')

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

    if scorer_type == 'voting_hard_mlps_svm_gradboost':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        mlp1 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp2 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp3 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp4 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)
        seed = np.random.randint(1, sys.maxsize)
        mlp5 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)

        svm1 = svm.SVC(gamma=0.001, C=100., verbose=True)
        gradboost = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

        clf = VotingClassifier(estimators=[  # ('gb', gb),
            # ('mlp', mlp),
            ('mlp', mlp1),
            ('mlp', mlp2),
            ('mlp', mlp3),
            ('mlp', mlp4),
            ('mlp', mlp5),
            ('svm', svm1),
            ('grad_boost', gradboost
             )
        ],  n_jobs=1,
            voting='hard')

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

    if scorer_type == 'voting_soft_svm_gradboost_logistic':
        svm2 = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', verbose=True)
        log_reg = logistic.LogisticRegression()
        gradboost = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)

        clf = VotingClassifier(estimators=[  # ('gb', gb),
            ('svm', svm2),
            ('grad_boost', gradboost),
            ('logisitc_regression', log_reg)
        ],  n_jobs=1,
            voting='soft')

    if scorer_type == 'voting_hard_mlp_riedel':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        mlp1 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)

        seed = np.random.randint(1, sys.maxsize)
        mlp2 = MultiThreadingFeedForwardMLP(n_classes=4, batch_size=188, hm_epochs=70, keep_prob_const=1.0, optimizer='adam',
                                            learning_rate=0.001, step_decay_LR=True, weight_init='sqrt_n', bias_init=0.001,
                                            hidden_layers=(362, 942, 1071, 870, 318, 912, 247), activation_function='relu',
                                            save_folder=save_folder, seed=seed)

        riedel1 = riedel_mlp(save_folder=save_folder+"1/")
        riedel2 = riedel_mlp(save_folder=save_folder+"2/")

        clf = VotingClassifier(estimators=[
            ('mlp', mlp1),
            ('riedel', riedel1),
            ('mlp', mlp2),
            ('riedel', riedel2)
        ],  n_jobs=1,
            voting='hard')

    if scorer_type == 'voting_hard_riedel':
        riedel1 = riedel_mlp(save_folder=save_folder+"1/")
        riedel2 = riedel_mlp(save_folder=save_folder+"2/")
        riedel3 = riedel_mlp(save_folder=save_folder+"3/")
        riedel4 = riedel_mlp(save_folder=save_folder+"4/")
        riedel5 = riedel_mlp(save_folder=save_folder+"5/")

        clf = VotingClassifier(estimators=[
            ('riedel', riedel1),
            ('riedel', riedel2),
            ('riedel', riedel3),
            ('riedel', riedel4),
            ('riedel', riedel5),
        ],  n_jobs=1,
            voting='hard')

    #Taken from Benjamins LSTM
    # I pass a random seed through the get_estimator() function => set fixed/random/anything
    # both models need around 5,2 GB GPU memory, so adjust gpu_memory_fraction accordingly
    if scorer_type == 'single_f_ext_LSTM_att_no_cw':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        #if features != None and isinstance(features, list):
        #    clf = single_f_ext_LSTM_att(epochs=100, batch_size=128, param_dict=features[0], lr=0.001, optimizer="adam", seed=seed, min_epoch=150, use_class_weights=False, gpu_memory_fraction=0.3)
        clf = single_f_ext_LSTM_att(epochs=100, batch_size=128, param_dict="single_flat_LSTM_50d_100", lr=0.001, optimizer="adam", seed=seed, min_epoch=150, use_class_weights=False, gpu_memory_fraction=0.3, save_folder=save_folder)

    if scorer_type == 'single_f_ext_LSTM_no_cw' or scorer_type == 'stackLSTM':
        import sys
        seed = np.random.randint(1, sys.maxsize)
        #if features != None and isinstance(features, list):
        #    clf = single_f_ext_LSTM(epochs=100, batch_size=128, param_dict=features[0], lr=0.001, optimizer="adam", seed=seed, min_epoch=150, use_class_weights=False, gpu_memory_fraction=0.3)
        clf = single_f_ext_LSTM(epochs=100, batch_size=128, param_dict="single_flat_LSTM_50d_100", lr=0.001, optimizer="adam", seed=seed, min_epoch=150, use_class_weights=False, gpu_memory_fraction=0.3, save_folder=save_folder)

    if scorer_type == 'sBalancedBagging':
        from sklearn.tree import DecisionTreeClassifier
        from imblearn.ensemble import BalancedBaggingClassifier
        clf = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),ratio='auto', replacement=False,random_state=0)

    return clf

