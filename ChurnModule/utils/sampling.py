from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from .log import get_logger

logger = get_logger(__name__)


class SamplingMethods:
    def __init__(self, X, y):
        """
        A custom class to perform various sampling methods

        Args:
            X (pd.DataFrame): Predictor variable
            y (pd.Series ot index): target variable
        """
        self.X = X
        self.y = y

    def sampling_technique(self, sampling_method=None, random_state=56, rus_sampling_ratio=0.5,
                           smote_sampling_ratio=0.5, smoterus_sampling_ratio=0.5, smote_k_neighbors=5,
                           categorical_features_index=None, smotenc_sampling_ratio=0.5, smotenc_k_neighbors=5,
                           smotesvm_sampling_ratio=0.5, smotesvm_k_neighbors=5, smotesvm_m_neighbors=10,
                           smotesvm_stepsize=0.5, adasyn_sampling_ratio=0.25, adasyn_k_neighbors=5):
        """
        The function performs various sampling techniques
        \\ Please refer for link for detailed documentation: https://imbalanced-learn.readthedocs.io/en/stable/api.html
        The sampling techniques implemented are:
        1. Random under sampling
        2. SMOTE-NC
        3. ADASYN
        4. SMOTE
        5. SVMSMOTE
        Only method 1 and method 2 are applicable for categorical data

        Args:
            sampling_method (str or None): Enter sampling method. Currently implemented ("RandomUnderSample",
                                            "SmoteUnderSample","SmoteNC", "SmoteSVM", "ADASYN")
            random_state (int): random seed for reproducibility
            rus_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority class
                                        over the number of samples in the majority class after resampling
            smote_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority
                                          class over the number of samples in the majority class after resampling.
            smoterus_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority
                                            class over the number of samples in the majority class after resampling
            smote_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
            categorical_features_index (int or list): index for categorical features
            smotenc_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority
                                          class over the number of samples in the majority class after resampling
            smotenc_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
            smotesvm_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority
                                          class over the number of samples in the majority class after resampling
            smotesvm_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
            smotesvm_m_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
            smotesvm_stepsize (int): Step size when extrapolating
            adasyn_sampling_ratio (int): It corresponds to the desired ratio of the number of samples in the minority
                                         class over the number of samples in the majority class after resampling
            adasyn_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        if sampling_method is None or sampling_method not in ["RandomUnderSample", "SmoteUnderSample",
                                                              "SmoteNC", "SmoteSVM", "ADASYN"]:
            raise NotImplementedError("Please enter sampling methods")

        elif sampling_method == "RandomUnderSample":
            X_res, y_res = self.random_under_sample(random_state, rus_sampling_ratio)

        elif sampling_method == "SmoteUnderSample":
            X_res, y_res = self.smote_under_sample(random_state, smote_sampling_ratio, smoterus_sampling_ratio,
                                                   smote_k_neighbors)

        elif sampling_method == "SmoteSVM":
            X_res, y_res = self.smote_svm(random_state, smotesvm_sampling_ratio, smotesvm_k_neighbors,
                                          smotesvm_m_neighbors, smotesvm_stepsize)
        elif sampling_method == "SmoteNC":
            X_res, y_res = self.smote_nc(random_state=random_state,
                                         categorical_features_index=categorical_features_index,
                                         smotenc_k_neighbor=smotenc_k_neighbors,
                                         smotenc_sampling_ratio=smotenc_sampling_ratio)

        elif sampling_method == "ADASYN":
            X_res, y_res = self.adasyn(random_state, adasyn_sampling_ratio, adasyn_k_neighbors)

        return X_res, y_res

    def random_under_sample(self, random_state, rus_sampling_ratio):
        """
        The function performs random under sampling and returns sampled data

        Args:
            random_state (int): random seed for reproducibility
            rus_sampling_ratio: It corresponds to the desired ratio of the number of samples in the minority
            class over the number of samples in the majority class after resampling

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        rus = RandomUnderSampler(random_state=random_state, sampling_strategy=rus_sampling_ratio)
        X_res, y_res = rus.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using random undersampling shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_under_sample(self, random_state, smote_sampling_ratio,
                           smoterus_sampling_ratio, smote_k_neighbors):
        """
        The function performs random under sampling and returns sampled data

        Args:
            random_state (int): random seed for reproducibility
            smote_sampling_ratio (float): It corresponds to the desired ratio of the number of samples in the minority
                                          class over the number of samples in the majority class after resampling.
            smoterus_sampling_ratio (float): It corresponds to the desired ratio of the number of samples in the
                                             minority
            smote_k_neighbors (int):If int, number of nearest neighbours to used to construct synthetic samples

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        over = SMOTE(sampling_strategy=smote_sampling_ratio, k_neighbors=smote_k_neighbors, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=smoterus_sampling_ratio, random_state=random_state)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE undersampling shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_nc(self, random_state, categorical_features_index, smotenc_k_neighbor,
                 smotenc_sampling_ratio):
        """
        The function performs random under sampling and returns sampled data

        Args:
            random_state (int): random seed for reproducibility
            categorical_features_index (int or list): array of indices specifying the categorical features
            smotenc_k_neighbor (int):  number of nearest neighbours to used to construct synthetic samples.
            smotenc_sampling_ratio (float): it corresponds to the desired ratio of the number of samples in the minority
                                            class over the number of samples in the majority class after resampling.

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        smote_oversample = SMOTENC(sampling_strategy=smotenc_sampling_ratio,
                                   random_state=random_state,
                                   k_neighbors=smotenc_k_neighbor,
                                   categorical_features=categorical_features_index)
        X_res, y_res = smote_oversample.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE NC shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_svm(self, random_state, smotesvm_sampling_ratio, smotesvm_k_neighbors,
                  smotesvm_m_neighbors, smotesvm_stepsize):
        """
        The function performs random under sampling and returns sampled data

        Args:
            random_state (int): random seed for reproducibility
            smotesvm_sampling_ratio (float): it corresponds to the desired ratio of the number of samples in the
                                            minority class over the number of samples in the majority class after
                                            resampling.
            smotesvm_k_neighbors (int): number of nearest neighbours to used to construct synthetic samples.
            smotesvm_m_neighbors (int): number of nearest neighbours to use to determine if a minority sample is in
                                        danger
            smotesvm_stepsize (float): Step size when extrapolating

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        oversample = SVMSMOTE(sampling_strategy=smotesvm_sampling_ratio,
                              random_state=random_state,
                              k_neighbors=smotesvm_k_neighbors,
                              m_neighbors=smotesvm_m_neighbors,
                              out_step=smotesvm_stepsize)
        X_res, y_res = oversample.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using SMOTE SVM shape %s' % Counter(y_res))
        return X_res, y_res

    def adasyn(self, random_state, adasyn_sampling_ratio, adasyn_k_neighbors):
        """
        The function performs random under sampling and returns sampled data

        Args:
            random_state (int): random seed for reproducibility
            adasyn_sampling_ratio (float):  It corresponds to the desired ratio of the number of samples in the minority
                                        class over the number of samples in the majority class after resampling
            adasyn_k_neighbors (int): number of nearest neighbours to used to construct synthetic samples.

        Returns:
            X_res (pd.DataFrame): sampled predictor data
            y_res (series) sampled target

        """
        ada = ADASYN(sampling_strategy=adasyn_sampling_ratio,
                     random_state=random_state,
                     n_neighbors=adasyn_k_neighbors)
        X_res, y_res = ada.fit_resample(self.X, self.y)
        logger.info('Resampled dataset using ADASYN shape %s' % Counter(y_res))
        return X_res, y_res
