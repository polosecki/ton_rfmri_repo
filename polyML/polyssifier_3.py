'''
This is a heavily edited and refactored version
of Sergey Plis' polyssifier
(https://github.com/pliz/polyssifier/blob/master/polyssifier.py)

It contains extra functionaly added by Pouya Bashivan and Pablo Polosecki
'''

__author__ = ["Sergey Plis"]
__copyright__ = ["Copyright 2015, Mind Research Network"]
__credits__ = ["Sergey Plis, Devon Hjelm, Alvaro Ulloa"]
__licence__ = "3-clause BSD"
__email__ = "splis@gmail.com"
__maintainer__ = "Sergey Plis"

__USEJOBLIB__ = False

from glob import glob
import logging

import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
from os import path
import pandas as pd
import pickle
import scipy.io
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from lightning.classification import (CDClassifier, SDCAClassifier,
                                      AdaGradClassifier, SAGAClassifier)
from pyglmnet import GLM
from glmnet import LogitNet
import copy
from scipy.io import loadmat
import sys
if sys.version_info[0] < 3:
    import TON_tools3 as TON_tools
else:
    from . import TON_tools3 as TON_tools
if __USEJOBLIB__:
    from joblib.pool import MemmapingPool as Pool
else:
    from multiprocessing import Pool


FONT_SIZE = 22
mpl.rcParams['pdf.fonttype']=42
mpl.use("Agg")

# please set this number to no more than the number of cores on the machine you're
# going to be running it on but high enough to help the computation
PROCESSORS = 42

np.random.seed(6606)


def rank_vars(xTrain, yTrain, scoreFunc, covariate_detrend_params=None):
    """
    ranks variables using the supplied scoreFunc.
    Inputs:
        xTrain: training features matrix
        yTrain: training labels vector
        scoreFunc: function used to rank features (pearsonr or mutual_info_score)
        covariate_detrend_params: dict or None. If dict apply detrending using
        those parameters
    Output:
        returns the ranked list of features indices
    """
    if covariate_detrend_params: #In this case, Xtrain are train indices to apply
        detrender = TON_tools.TON_feature_detrender(**covariate_detrend_params)
        idx_train, idx_test = xTrain
        detrender.fit(idx_train, idx_test) # X train are indices of the filtered data matrix,
        #idx_test is provided to EXPLICITPLY REMOVE THEM FROM THE TRAIN DATA
        # this is accomodated by the fit method of the detrender
        usedX = detrender.transform(idx_train)
    else:
        usedX = xTrain

    funcsDic = {
        'pearsonr': [np.arange(usedX.shape[1]), 1],
        'mutual_info_score': np.arange(usedX.shape[0]),
        'ttest_ind': [np.arange(usedX.shape[1]), 1],
        }

    scores = list()
    for feat in np.arange(usedX.shape[1]):
        if scoreFunc.__name__ == 'pearsonr':
            scores.append(scoreFunc(usedX[:, feat], yTrain))
        elif scoreFunc.__name__ == 'ttest_ind':
            scores.append(scoreFunc(usedX[yTrain == 1, feat], usedX[yTrain==-1, feat]))

    scores = np.asarray(scores)
    pvals = scores[funcsDic[scoreFunc.__name__]]
    dtype = [('index', int), ('p-val',float)]
    ind_and_pvals = [x for x in enumerate(pvals)]
    sortedIndices = [i[0] for i in np.sort(np.array(ind_and_pvals,dtype=dtype),
                     order='p-val')]
    return sortedIndices

class SupervisedStdScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = StandardScaler()
        self.__subscaleru = StandardScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.mean_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (X-tile_center) / tile_scale
        return X_transf

class SupervisedRobustScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = RobustScaler()
        self.__subscaleru = RobustScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.center_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (4./3) * (X-tile_center) / tile_scale
        return X_transf
    #If data is normally distributed, it will have unit variance


class Ranker(object):
    """
    Class version of univariate ranking, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        ranking function: the ranking function to give to rank_vars
        rank_vars: the rank_vars function
        fp: list of fold train-test pairs
        covariate_detrend_params: dict or None. default: None
            If dict, detrend is applied with these parameters
    """
    def __init__(self, data, labels, ranking_function, rank_vars=rank_vars,
                 covariate_detrend_params=None, give_idx_to_ranker=False):
        self.data = data
        self.labels = labels
        self.rf = ranking_function
        self.rank_vars = rank_vars
        self.covariate_detrend_params = covariate_detrend_params
        if covariate_detrend_params:
            give_idx_to_ranker = True
        self.give_idx_to_ranker = give_idx_to_ranker

    def __call__(self, fp):
        if self.give_idx_to_ranker:
            rv = self.rank_vars(fp, self.labels[fp[0]],
                                self.rf,
                                covariate_detrend_params=self.covariate_detrend_params)
        else:
            rv = self.rank_vars(self.data[fp[0], :], self.labels[fp[0]],
                                self.rf,
                                covariate_detrend_params=self.covariate_detrend_params)
        return rv

def get_rank_per_fold(data, labels, fold_pairs, ranking_function=ttest_ind,
                      save_path=None,load_file=True,
                      parallel=True, covariate_detrend_params=None):
    '''
    Applies rank_vars to each test set in list of fold pairs
    Inputs:
        data: array
            features for all samples
        labels: array
            label vector of each sample
        fold_pair: list
            list pairs of index arrays containing train and test sets
        ranking_function: function object, default: ttest_ind
            function to apply for ranking features
        ranking_function: function
            ranking function to use, default: ttest_ind
        save_path: dir to load and save ranking files
        load_file: bool
            Whether to try to load an existing file, default: True
        parallel: bool
            True if multicore processing is desired, default: True
        covariate_detrend_params: Dict or None, default: None
            If Dict, apply covariate detreding with these dict of parameters
    Outputs:
        rank_per_fod: list
            List of ranked feature indexes for each fold pair
    '''
    file_loaded = False
    if load_file:
        if isinstance(save_path, str):
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))
            try:
                rd = scipy.io.loadmat(fname, mat_dtype = True)
                rank_per_fold = rd['rank_per_fold'].tolist()
                file_loaded = True
            except:
                pass
        else:
            print('No rank file path: Computing from scratch without saving')
    if not file_loaded:
        if not parallel:
            rank_per_fold = []
            gg=0
            for fold_pair in fold_pairs:
                print('Fold pair {}'.format(gg))
                gg += 1
                if covariate_detrend_params:
                    print
                    rankedVars = rank_vars(fold_pair,
                                           labels[fold_pair[0]],
                                           ranking_function,
                                           covariate_detrend_params)
                else:
                    rankedVars = rank_vars(data[fold_pair[0], :],
                                           labels[fold_pair[0]],
                                           ranking_function)
                rank_per_fold.append(rankedVars)
        else:
            pool = Pool(processes=min(len(fold_pairs), PROCESSORS))
            rank_per_fold = pool.map(Ranker(data, labels, ranking_function,
                                            rank_vars,
                                            covariate_detrend_params),
                                    fold_pairs)
            pool.close()
            pool.join()
        if isinstance(save_path, str):
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))
            with open(fname, 'wb') as f:
                scipy.io.savemat(f, {'rank_per_fold': rank_per_fold})
    return rank_per_fold

def make_classifiers(NAMES) :
    """Function that makes classifiers each with a number of folds.

    Returns two dictionaries for the classifiers and their parameters, using
    `data_shape` and `ksplit` in construction of classifiers.

    Parameters
    ----------
    data_shape : tuple of int
        Shape of the data.  Must be a pair of integers.
    ksplit : int
        Number of folds.

    Returns
    -------
    classifiers: dict
        The dictionary of classifiers to be used.
    params: dict
        A dictionary of list of dictionaries of the corresponding
        params for each classifier.
    """

#    if len(data_shape) != 2:
#        raise ValueError("Only 2-d data allowed (samples by dimension).")

    classifiers = {
        "Chance": DummyClassifier(strategy="most_frequent"),
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": LinearSVC(penalty='l2', C=1,# probability=True,
                          class_weight='balanced'),
        "RBF SVM": SVC(gamma=2, C=1, probability=True),
        "Decision Tree": DecisionTreeClassifier(max_depth=None,
                                                max_features="auto"),
        "Random Forest": RandomForestClassifier(max_depth=None,
                                                n_estimators=50,
                                                max_features="auto",
                                                n_jobs=PROCESSORS),
        "Logistic Regression": LogisticRegression(penalty='l1',
                                                   class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "LDA": LDA(),
        "SGD_logL1": SGDClassifier(random_state=1952,loss='log', average = 3,
                                  penalty='l1',
                                  alpha=1e-3,
                                  class_weight='balanced'),
        "SGD_log_elastic": SGDClassifier(random_state=1952,loss='log',
                                          class_weight='balanced',
                                          alpha=1e-3,
                                          average = 3,
                                          penalty='elasticnet'),
        "SGD_SVM_elastic": SGDClassifier(random_state=1952,loss='hinge',
                                          class_weight='balanced',
                                          average = 3,
                                          alpha=1e-3,
                                          penalty='elasticnet'),

        "CGC_log_L1": CDClassifier(penalty="l1",
                   loss="log",
                   multiclass=False,
                   max_iter=200,
                   C=1,
                   tol=1e-3),
        "SDCA_SVM_elastic": SDCAClassifier(
                   loss="hinge",
                   max_iter=200,
                   tol=1e-3),
        "AdaBoostSVM": AdaBoostClassifier(SVC(C=1,
                                              class_weight='balanced',
                                              kernel='linear',
                                              probability=False),
                                           n_estimators=10,
                                           algorithm='SAMME'),
        "BaggingSVM": BaggingClassifier(base_estimator=LinearSVC(class_weight='balanced',
                                                                 loss='hinge'),
                                        n_estimators=21),
        "AdaGrad_log_elastic": AdaGradClassifier(loss='log'),
        "SAGA_log_elastic": SAGAClassifier(loss='log',penalty='l1'),
        "pyglm_logistic": GLM(distr='binomial',
                              solver='cdfast'),
        "glmnet_logistic": LogitNet(standardize=False)
        }

    params = {
        "Chance": {},
        "Nearest Neighbors": {"n_neighbors": [1, 5, 10, 20]},
        "Linear SVM": {"C": np.logspace(-5,3,8),
                       "loss":['hinge']},# 'squared_hinge']},
        "RBF SVM": {"kernel": ["rbf"],
                     "gamma": np.logspace(-2, 0, 6).tolist() + \
                              np.logspace(0,1,5)[1:].tolist(),
                     "C": np.logspace(-2, 2, 5).tolist()},
        "Decision Tree": {},
        "Random Forest": {"max_depth": np.round(np.logspace(np.log10(2), \
                                       1.2, 6)).astype(int).tolist()},
        "Logistic Regression": {"C": np.logspace(-2, 3, 1).tolist()},
        "Naive Bayes": {},
        "LDA": {},
        "SGD_logL1": {"alpha": np.logspace(-5, 2, 1)},
        "SGD_log_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "SGD_SVM_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "CGC_log_L1": {"alpha": np.logspace(-5, 2, 1)},
        "SDCA_SVM_elastic": {"alpha": np.logspace(-4, 4, 1),
                             "l1_ratio": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)])},
        "AdaBoostSVM": {'base_estimator__C': np.logspace(-1,4,5)},  # ATTENTION, MAKE THIS REASONABLE AGAIN
        "BaggingSVM": {'base_estimator__C': np.logspace(-5,2,8)},
        "AdaGrad_log_elastic": {"alpha": np.logspace(-4, 4, 1), #np.logspace(-4, 4, 5),
                             "l1_ratio": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)])},
        "SAGA_log_elastic": {"alpha": np.logspace(-4, 4, 5), #np.logspace(-2, 2, 3), #np.logspace(-4, 4, 5),
                             "beta": np.logspace(-2, 2, 1)}, #np.logspace(-2, 2, 3)},
        "pyglm_logistic": {"alpha": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)]),
                           "reg_lambda": np.logspace(-4, 4, 5)},
        "glmnet_logistic": {"alpha": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)]),
                            "lambda_path": [w for w in np.logspace(-4, 4, 5)[:,np.newaxis]]}
            }
    out_classifiers = {cname: classifiers[cname] for cname in NAMES}
    out_params = {cname: params[cname] for cname in NAMES}
    logging.info("Using classifiers %r with params %r" % (out_classifiers,
                                                         out_params))
    return classifiers, params


class per_split_classifier(object):
    """
    Class version of classify function, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        classifier: classifier object to use
        numTopVars: list of top variables to use
        zipped_ranks_n_fp: zipped list 2-tuple with ranked vars and train-test
                           indices
        fp: a single train-test pair
    """
    def __init__(self, data, labels, classifier, numTopVars,
                 covariate_detrend_params=None):
        self.data = data
        self.labels = labels
        self.clf = classifier
        self.numTopVars = numTopVars
        self.covariate_detrend_params = covariate_detrend_params

    def __call__(self, zipped_ranks_n_fp):
        rankedVars, fp = zipped_ranks_n_fp
        confMats = []
        totalErrs = []
        fitted_classifiers = []
        predictions = []
        cont_preds = []
        for numVars in self.numTopVars:
            if self.covariate_detrend_params:
               self.covariate_detrend_params['rankedVars'] =  rankedVars
               self.covariate_detrend_params['numVars'] = numVars
            classify_output = classify(self.data[:, rankedVars[:numVars]],
                                       self.labels, fp, self.clf,
                                       covariate_detrend_params=self.covariate_detrend_params)
            confMats.append(classify_output[0])
            totalErrs.append(classify_output[1])
            fitted_classifiers.append(classify_output[2])
            predictions.append(classify_output[3])
            cont_preds.append(classify_output[4])
        return confMats, totalErrs, fitted_classifiers, predictions, cont_preds


def get_score(data, labels, fold_pairs, name, model, param, numTopVars,
              rank_per_fold=None, parallel=True, rand_iter=-1,
              covariate_detrend_params=None,
              provide_continuous_output=True):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array_like
        Data from which to derive score.
    labels: array_like or list
        Corresponding labels for each sample.
    fold_pairs: list of pairs of array_like
        A list of train/test indicies for each fold
        dhjelm(Why can't we just use the KFold object?)
    name: str
        Name of classifier.
    model: WRITEME
    param: WRITEME
        Parameters for the classifier.
    parallel: bool
        Whether to run folds in parallel. Default: True

    Returns
    -------
    classifier: WRITEME
    allConfMats: Confusion matrix for all folds and all variables sets and best performing parameter set
                 ([numFolds, numVarSets])
    """
    assert isinstance(name, str)
    logging.info("Classifying %s" % name)
    ksplit = len(fold_pairs)
#    if name not in NAMES:
#        raise ValueError("Classifier %s not supported. "
#                         "Did you enter it properly?" % name)

    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)
    if "SGD" in name:
        param["n_iter"] = [25]  # [np.ceil(10**3 / len(fold_pairs[0][0]))]
    classifier = get_classifier(name, model, param, rand_iter=rand_iter)

    if name == "RBF SVM": #This doesn't use labels, but looks as ALL data
        logging.info("RBF SVM requires some preprocessing."
                    "This may take a while")
        #Euclidean distances between samples
        dist = pdist(StandardScaler().fit(data), "euclidean").ravel()
        #dist = pdist(RobustScaler().fit_transform(data), "euclidean").ravel()
        #Estimates for sigma (10th, 50th and 90th percentile)
        sigest = np.asarray(np.percentile(dist,[10,50,90]))
        #Estimates for gamma (= -1/(2*sigma^2))
        gamma = 1./(2*sigest**2)
        #Set SVM parameters with these values
        param = [{"kernel": ["rbf"],
                  "gamma": gamma.tolist(),
                  "C": np.logspace(-2,2,5).tolist()}]
    # if name not in ["Decision Tree", "Naive Bayes"]:
    if param:
        if hasattr(classifier,'param_grid'):
        # isinstance(classifier, GridSearchCV):
            N_p = np.prod([len(l) for l in param.values()])
        elif isinstance(classifier, RandomizedSearchCV):
            N_p = classifier.n_iter
    else:
        N_p = 1
#    is_cv = isinstance(classifier, GridSearchCV) or \
#            isinstance(classifier, RandomizedSearchCV)
#    print('Name: {}, ksplit: {}, N_p: {}'.format(name, ksplit, N_p))
    if (not parallel) or \
    (name == "Random Forest") or ("SGD" in name):
    # or ksplit <= N_p:
        logging.info("Attempting to use grid search...")
        classifier.n_jobs = PROCESSORS
        # classifier.pre_dispatch = 1 # np.floor(PROCESSORS/24)
        allConfMats = []
        allTotalErrs = []
        allFittedClassifiers = []
        allPredictions = []
        allContPreds = []
        for i, fold_pair in enumerate(fold_pairs):
            confMats = []
            totalErrs = []
            fitted_classifiers = []
            predictions = []
            cont_preds = []
            logging.info("Classifying a %s the %d-th out of %d folds..."
                   % (name, i+1, len(fold_pairs)))
            if rank_per_fold is not None:
                rankedVars = np.squeeze(rank_per_fold)[i]
            else:
                rankedVars = np.arange(data.shape[1])
            for numVars in numTopVars:
                logging.info('Classifying for top %i variables' % numVars)
                if covariate_detrend_params:
                   covariate_detrend_params['rankedVars'] =  rankedVars
                   covariate_detrend_params['numVars'] = numVars
                classify_output = classify(data[:, rankedVars[:numVars]],
                                           labels,
                                           fold_pair,
                                           classifier,
                                           covariate_detrend_params=covariate_detrend_params)
                confMat, totalErr, fitted_classifier, prediction, cont_pred\
                    = classify_output
                confMats.append(confMat)
                totalErrs.append(totalErr)
                fitted_classifiers.append(fitted_classifier)
                predictions.append(prediction)
                cont_preds.append(cont_pred)
            # recheck the structure of area and fScore variables
            allConfMats.append(confMats)
            allTotalErrs.append(totalErrs)
            allFittedClassifiers.append(fitted_classifiers)
            allPredictions.append(predictions)
            allContPreds.append(cont_preds)
    else:
        classifier.n_jobs = PROCESSORS
        logging.info("Multiprocessing folds for classifier {}.".format(name))
        pool = Pool(processes=min(ksplit, PROCESSORS))
        out_list = pool.map(per_split_classifier(data, labels, classifier,
                                                 numTopVars,
                                                 covariate_detrend_params=covariate_detrend_params),
                            zip(rank_per_fold, fold_pairs))
        pool.close()
        pool.join()
        #allConfMats = [el[0] for el in out_list]
        #allTotalErrs = [el[1] for el in out_list]
        #allFittedClassifiers = [el[2] for el in out_list]
        allConfMats, allTotalErrs, allFittedClassifiers, allPredictions, allContPreds\
            = tuple(zip(*out_list))
    return classifier, allConfMats, allTotalErrs, allFittedClassifiers, allPredictions, allContPreds

def get_classifier(name, model, param, rand_iter=-1):
    """
    Returns the classifier for the model.

    Parameters
    ----------
    name: str
        Classifier name.
    model: WRITEME
    param: WRITEME
    data: array_like, optional

    Returns
    -------
    WRITEME
    """
    assert isinstance(name, str)
    if param: # Do grid search only if parameter list is not empty
        N_p = np.prod([len(l) for l in param.values()])
        if (N_p <= rand_iter) or rand_iter<=0:
            logging.info("Using grid search for %s" % name)
            model = GridSearchCV(model, param, cv=5, scoring="accuracy",
                                 n_jobs=PROCESSORS)
        else:
            logging.info("Using random search for %s" % name)
            model = RandomizedSearchCV(model, param, cv=5, scoring="accuracy",
                                 n_jobs=PROCESSORS, n_iter=rand_iter)
    else:
        logging.info("Not using grid search for %s" % name)
    return model

def classify(data, labels, train_test_idx, classifier=None,
             covariate_detrend_params=None):

    """
    Classifies given a fold and a model.

    Parameters
    ----------
    data: array_like
        2d matrix of observations vs variables
    labels: list or array_like
        1d vector of labels for each data observation
    (train_idx, test_idx) : list
        set of indices for splitting data into train and test
    classifier: sklearn classifier object
        initialized classifier with "fit" and "predict_proba" methods.

    Returns
    -------
    WRITEME
    """

    assert classifier is not None, "Why would you pass not classifier?"
    train_idx, test_idx = train_test_idx
    # Perform detrending:
    if covariate_detrend_params:
        detrender = TON_tools.TON_feature_detrender(**covariate_detrend_params)
        detrender.fit(train_idx, test_idx) # test_idx provided to be EPXLICITLY removed
        # from the fitting procedure
        clean_data = detrender.transform(np.union1d(train_idx, test_idx))
    else:
        clean_data = data

    # Data scaling based on training set
    scaler = SupervisedStdScaler() #SupervisedRobustScaler()  #  #
    # scaler = StandardScaler()
    scaler.fit(clean_data[train_idx,:], labels[train_idx], label=-1)
    scaler.fit(clean_data[train_idx, :], labels[train_idx])
    data_train = scaler.transform(clean_data[train_idx, :])
    data_test = scaler.transform(clean_data[test_idx, :])
    #from IPython.terminal.debugger import TerminalPdb; TerminalPdb().set_trace()

    try:
        classifier.fit(data_train, labels[train_idx])

        predictions = classifier.predict(data_test)

        confMat = confusion_matrix(labels[test_idx],
                                   predictions)
        cont_prediction = np.nan * np.zeros(predictions.shape)
        try:
            cont_prediction = classifier.predict_proba(data_test)
        except AttributeError:
            try:
                cont_prediction = classifier.decision_function(data_test)
            except AttributeError:
                pass
        if len(cont_prediction.shape) > 1:
            cont_prediction = cont_prediction[:,-1]
        if confMat.shape == (1, 1):
            if all(labels[test_idx] == -1):
                confMat = np.array([[confMat[0], 0], [0, 0]],
                                   dtype=confMat.dtype)
            else:
                confMat = np.array([[0, 0], [0, confMat[0]]],
                                   dtype=confMat.dtype)
        confMatRate = confMat / np.tile(np.sum(confMat, axis=1).
                                        astype('float'), (2, 1)).transpose()
        totalErr = (confMat[0, 1] + confMat[1, 0]) / float(confMat.sum())
        # if type(classifier) not in [type(None), DummyClassifier]:
        if hasattr(classifier, 'param_grid'):
        # isinstance(classifier, GridSearchCV) or \
        #    isinstance(classifier, RandomizedSearchCV):
                fitted_model = classifier.best_estimator_
        else:
                fitted_model = copy.copy(classifier)


        return confMatRate, totalErr, fitted_model, predictions, cont_prediction
    except np.linalg.linalg.LinAlgError:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan, None, np.nan, np.nan


def load_data(data_file, data_pattern='*.mat'):
    """
    Loads the data from multiple sources if provided.

    Parameters
    ----------
    data_file: str
    data_pattern: str

    Returns
    -------
    data: array_like
    """

    dataMat = scipy.io.loadmat(data_file, mat_dtype = True)
    data = dataMat['data']

    logging.info("Data loading complete. Shape is %r" % (data.shape,))
    return data[:, :-1], data[:, -1], data_file

def load_labels(source_dir, label_pattern):
    """
    Function to load labels file.

    Parameters
    ----------
    source_dir: str
        Source directory of labels
    label_pattern: str
        unix regex for label files.

    Returns
    -------
    labels: array_like
        A numpy vector of the labels.
    """

    logging.info("Loading labels from %s with pattern %s"
                % (source_dir, label_pattern))
    label_files = glob(path.join(source_dir, label_pattern))
    if len(label_files) == 0:
        raise ValueError("No label files found with pattern %s"
                         % label_pattern)
    if len(label_files) > 1:
        raise ValueError("Only one label file supported ATM.")
    labels = np.load(label_files[0]).flatten()
    logging.info("Label loading complete. Shape is %r" % (labels.shape,))
    return labels

def load_subject_list(data_file, source='matlab'):
    """
    provides the lsit of subjects

    Parameters
    ----------
    data_file: filename
    source: 'matlab' or 'python'

    Returns
    -------
    subject_list: list fo sujects in mat file
    """
    zz = loadmat(data_file)['fmri_subjects']
    if source == 'matlab':
        subject_list = np.array([s[0][0] for s in zz])
    elif source == 'python':
        try:
            tt = isinstance(zz[0], (str, unicode))
        except NameError:
            tt = isinstance(zz[0], str)
        if tt:
            subject_list = zz
        else:
            subject_list = np.array([str(s[0]) for s in zz[0]])
    subject_list = np.array([s.replace(' ','') for s in subject_list])
    return subject_list


def save_classifier_results(classifier_name, out_dir, allConfMats,
                            allTotalErrs):
    """
    saves the classifier results including TN, FN and total error.
    """

    # convert confusion matrix and total errors into numpy array
    tmpAllConfMats = np.array(allConfMats)
    tmpAllTotalErrs = np.array(allTotalErrs)
    # initialize mean and std variables
    TN_means = np.zeros(tmpAllConfMats.shape[1])
    TN_stds = np.zeros(tmpAllConfMats.shape[1])
    FN_means = np.zeros(tmpAllConfMats.shape[1])
    FN_stds = np.zeros(tmpAllConfMats.shape[1])
    total_means = np.zeros(tmpAllConfMats.shape[1])
    total_stds = np.zeros(tmpAllConfMats.shape[1])

    for j in range(tmpAllConfMats.shape[1]):
        tmpData = tmpAllConfMats[:, j, 0, 0]
        TN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        TN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllConfMats[:, j, 1, 0]
        FN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        FN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllTotalErrs[:, j]
        # Compute mean of std of non-Nan values
        total_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        total_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])

    with open(path.join(out_dir, classifier_name + '_errors.mat'), 'wb') as f:
        scipy.io.savemat(f, {'TN_means': TN_means,
                             'TN_stds': TN_stds,
                             'FN_means': FN_means,
                             'FN_stds': FN_stds,
                             'total_means': total_means,
                             'total_stds': total_stds,
                             })


def save_classifier_predictions_per_sample(classifier_name, out_dir,
                                           predictions, cont_predictions,
                                           fold_pairs, labels,
                                           subjects_per_run, ktop=-1):
    '''
    Construction of a data frame with labels, calssifier predictions and
    subject name of each sample
    '''
    test_idx = (fp[1] for fp in fold_pairs)
    idx_list = []
    fold_list = []
    pred_list = []
    cont_pred_list = []
    in_labels = []
    sample_subj = []
    for (fold_idx, (indx_from_fold, pred_from_fold, cont_pred_from_fold))\
            in enumerate(zip(test_idx, predictions, cont_predictions)):
        idx_list.extend(indx_from_fold.tolist())
        pred_list.extend(pred_from_fold[ktop].astype(int).tolist())
        cont_pred_list.extend(cont_pred_from_fold[ktop].tolist())
        in_labels.extend(labels[indx_from_fold].astype(int).tolist())
        sample_subj.extend(subjects_per_run[indx_from_fold].tolist())
        fold_list.extend([fold_idx] * len(indx_from_fold))
    label_pred_per_sample = pd.DataFrame(dict(zip(('sample', 'subjid',
                                                   'labels', 'prediction',
                                                   'continuous_prediction',
                                                   'test_fold'),
                                                  (idx_list, sample_subj,
                                                   in_labels, pred_list,
                                                   cont_pred_list,
                                                   fold_list)))
                                         ).set_index('sample')
    label_pred_per_sample.to_csv(path.join(out_dir, classifier_name +
                                           '_predictions.csv'))


def save_classifier_object(clf, FittedClassifiers, name, out_dir):
    if out_dir is not None:
        save_path = path.join(out_dir, name + '.pkl')
        logging.info("Saving classifier to %s" % save_path)
        classifier_dict = {'name': name,
                           'classifier': clf,
                           'FittedClassifiers': FittedClassifiers}
        with open(save_path, "wb") as f:
            pickle.dump(classifier_dict, f, protocol=2)


def save_combined_results(NAMES, dscore, totalErrs, numTopVars, out_dir, filebase):
    confMatResults = {name.replace(" ", ""): scores for name, scores in zip(NAMES, dscore)}
    confMatResults['topVarNumbers'] = numTopVars
    totalErrResults = {name.replace(" ", ""): errs for name, errs in zip(NAMES, totalErrs)}
    totalErrResults['topVarNumbers'] = numTopVars
    # save results from all folds
    # dscore is a matrix [classifiers, folds, #vars, 2, 2]
    dscore = np.asarray(dscore)
    totalErrs = np.asarray(totalErrs)
    with open(path.join(out_dir, filebase + '_dscore_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'dscore': dscore,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})

    with open(path.join(out_dir, filebase + '_errors_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'errors': totalErrs,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})
    # Save all results
    with open(path.join(out_dir, 'confMats.mat'),'wb') as f:
        scipy.io.savemat(f, confMatResults)
    with open(path.join(out_dir, 'totalErrs.mat'),'wb') as f:
        scipy.io.savemat(f, totalErrResults)


def plot_errors(NAMES,numTopVars, dscore=None, totalErrs=None,
                filename_base='', out_dir=None, compute_error=True,
                format_used='png'):
    ######################################
    # Plot Figures
    # Confusion matrix format is:
    #   TN  FP
    #   FN  TP
    # Plotting false-positive ratio
    cl = [(1., 0., 0.),
          (0., 1., 0.),
          (0., 0., 1.),
          (0., 0., 0.),
          (.5, .5, 0.),
          (.5, 0., .5),
          (0., .5, .5),
          (.9, .9, .1),
          (0., 1., 1.),
          (1., 0., 1.),
          (1., .7, .3),
          (.5, 1., .7),
          (.7, .3, 1.),
          (.3, .7, 1.),
          (.3, .1, .7),
          (1., .3, .7)]

    ax = pl.gca()
    if not compute_error:
        dscore = range(len(NAMES))
        totalErrs = range(len(NAMES))
    if dscore:
        dscore = np.asarray(dscore)
        # Plotting FP rate
        handles = []
        means = np.zeros(len(numTopVars))  # np.zeros(dscore.shape[2])
        ax.set_prop_cycle(color=cl)
        for i in range(dscore.shape[0]):
            if compute_error:
                for j in range(dscore.shape[2]):
                    tmpData = dscore[i, :, j, 0, 1]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    #stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else:
                name=NAMES[i]
                fn = path.join(out_dir, 'confMats.mat')
                dm = scipy.io.loadmat(fn)
                z=dm[name.replace(' ', '')]
                xfold_mean=np.nanmean(z,0)
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype = True)

                means = xfold_mean[:,0,1]
            handles.append(pl.errorbar(numTopVars, means, fmt='-o'))
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xscale('log')
        ax.set_ylabel('FP rate', fontsize=FONT_SIZE)
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        ax.set_ylim((0,1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0., prop={'size':14})
        pl.grid()

        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_FP.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
        pl.cla()


        handles = []
        # Plotting false-negative ratio
        means = np.zeros(len(numTopVars))
        # stds = np.zeros(dscore.shape[2])
        ax.set_prop_cycle(color=cl)
        for i in range(dscore.shape[0]):
            if compute_error:
                for j in range(dscore.shape[2]):
                    tmpData = dscore[i, :, j, 1, 0]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    #stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else:
                name=NAMES[i]
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype = True)
                means = err_dict['FN_means'].flatten()
            handles.append(pl.errorbar(numTopVars, means, fmt='-o'))
        ax = pl.gca()
        ax.set_xscale('log')
        ax.set_ylabel('FN rate', fontsize=FONT_SIZE)
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        ax.set_ylim((0,1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0.,prop={'size':14})
        pl.grid()

        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_FN.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
        pl.cla()

    if totalErrs:
        totalErrs = np.asarray(totalErrs)
        handles = []
        # Plotting total error
        means = np.zeros(len(numTopVars))
        # stds = np.zeros(totalErrs.shape[2])
        ax = pl.gca()
        ax.set_prop_cycle(color=cl)
        for i in range(totalErrs.shape[0]):
            if compute_error:
                for j in range(totalErrs.shape[2]):
                    tmpData = totalErrs[i, :, j]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    #stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else:
                name=NAMES[i]
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype = True)
                means = err_dict['total_means'].flatten()
            handles.append(pl.errorbar(numTopVars, means, fmt='-o'))
        ax = pl.gca()
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xscale('log')
        ax.set_facecolor('w')  # set_axis_bgcolor('w')
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_ylabel('Error rate', fontsize=FONT_SIZE)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        ax.set_ylim((0,1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0., prop={'size':14})
        pl.grid()

        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_total_errors.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
    pl.clf()


def plot_bootstrap_confusion_from_prediction_csv(classifier_name, out_dir,
                                                 n_resamples=1000,
                                                 do_chance=True, alpha=5,
                                                 chance_alpha=None,
                                                 save_fig=True,
                                                 format_used='png',
                                                 fig_title='accuracy',
                                                 replacement=True):
    import bootstrap_tools as bt
    if chance_alpha is None:
        chance_alpha = alpha
    prediction_df = pd.read_csv(path.join(out_dir, classifier_name +
                                          '_predictions.csv'),
                                index_col='sample').sort_index().copy()
    perf_dict = bt.provide_performance_distributions_from_df(prediction_df,
                                                             n_resamples=n_resamples)
    if do_chance:
        rand_dist_dict = bt.provide_performance_distributions_from_df(prediction_df,
                    n_resamples=n_resamples, shuffle=True, replacement=replacement)
    else:
        rand_dist_dict = {}
    fh, ax = bt.plot_bootstrap_distributions(perf_dict, rand_dist_dict,
                                             perf_alpha=alpha,
                                             chance_alpha=chance_alpha,
                                             fig_title=fig_title)
    fh.savefig(path.join(out_dir, classifier_name.replace(' ', '_') +
                         '_conf_mat.' + format_used), dpi=300,
               bbox_inches='tight', format=format_used)
    return

def plot_bootstrap_auc_from_prediction_csv(classifier_name, out_dir,
                                           n_resamples=1000,
                                           do_chance=True, alpha=5,
                                           save_fig=True,
                                           format_used='png',
                                           fig_title='auc',
                                           replacement=False):
    import bootstrap_tools as bt

    prediction_df = pd.read_csv(path.join(out_dir, classifier_name +
                                          '_predictions.csv'),
                                index_col='sample').sort_index().copy()
    if do_chance:
        rand_dist = bt.provide_auc_distributions_from_df(prediction_df,
                                                         n_resamples=n_resamples,
                                                         shuffle=True,
                                                         replacement=replacement)
    else:
        rand_dist = np.nan
    fh, ax = bt.plot_ROC_curve_from_df(prediction_df, rand_dist=rand_dist,
                                       alpha=alpha, n_resamples=n_resamples)

    fh.savefig(path.join(out_dir, classifier_name.replace(' ', '_') +
                         '_auc.' + format_used), dpi=300,
               bbox_inches='tight', format=format_used)
    return

def plot_bootstrap_confusion_from_csv_in_chunks(classifier_name, group_dict,
                                                out_dir,
                                                n_resamples=1000,
                                                do_chance=True, alpha=5,
                                                chance_alpha=None,
                                                save_fig=True,
                                                format_used='png',
                                                fig_title='accuracy',
                                                replacement=False):
    '''
    Group dict: dict with training group names as keys,
    testing index arrays as values
    '''

    import bootstrap_tools as bt
    if chance_alpha is None:
        chance_alpha = alpha
    prediction_df = pd.read_csv(path.join(out_dir, classifier_name +
                                          '_predictions.csv'),
                                index_col='sample').sort_index().copy()
    for group_name, group_idx in group_dict.items():  # items():
        perf_dict = bt.provide_performance_distributions_from_df(prediction_df.iloc[group_idx].copy(),
                                                                 n_resamples=n_resamples)
        if do_chance:
            rand_dist_dict = bt.provide_performance_distributions_from_df(prediction_df.iloc[group_idx].copy(),
                        n_resamples=n_resamples, shuffle=True, replacement=replacement)
        else:
            rand_dist_dict = {}
        fh, ax = bt.plot_bootstrap_distributions(perf_dict, rand_dist_dict,
                                                 perf_alpha=alpha,
                                                 chance_alpha=chance_alpha,
                                                 fig_title=fig_title)
        #from IPython.terminal.debugger import TerminalPdb; TerminalPdb().set_trace()
        fh.savefig(path.join(out_dir, classifier_name.replace(' ', '_') +
                             '_conf_mat_' + group_name +'.' + format_used),
                   dpi=300, bbox_inches='tight', format=format_used)
    return


def plot_bootstrap_auc_from_prediction_csv_in_chunks(classifier_name,
                                                     group_dict,
                                                     out_dir,
                                                     n_resamples=1000,
                                                     do_chance=True, alpha=5,
                                                     save_fig=True,
                                                     format_used='png',
                                                     fig_title='auc',
                                                     replacement=False):
    '''
    Group dict: dict with training group names as keys,
    testing index arrays as values
    '''

    import bootstrap_tools as bt

    prediction_df = pd.read_csv(path.join(out_dir, classifier_name +
                                          '_predictions.csv'),
                                index_col='sample').sort_index().copy()
    for group_name, group_idx in group_dict.items():

        if do_chance:
            rand_dist = bt.provide_auc_distributions_from_df(prediction_df.iloc[group_idx].copy(),
                                                             n_resamples=n_resamples,
                                                             shuffle=True,
                                                             replacement=replacement)
        else:
            rand_dist = np.nan
        fh, ax = bt.plot_ROC_curve_from_df(prediction_df.iloc[group_idx].copy(),
                                           rand_dist=rand_dist,
                                           alpha=alpha, n_resamples=n_resamples)

        fh.savefig(path.join(out_dir, classifier_name.replace(' ', '_') +
                             '_auc_' + group_name + '.' + format_used), dpi=300,
                   bbox_inches='tight', format=format_used)
    return
