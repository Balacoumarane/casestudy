import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.model_selection import RandomizedSearchCV, KFold
from ..utils import get_logger
import pickle

logger = get_logger(__name__)


class XGBoostModel:

    def __init__(self, trainX=None, trainy=None, testX=None, testy=None, class_weights=None):
        """
        The function initalise data for training
        Args:
            trainX (pd.DataFrame): Training data for modelling
            trainy (pd.Series): Target variable  for training
            testX (pd.DataFrame): Testing data for modelling
            testy (pd.Series): Target variable for testing
            class_weights (int or None): The ratio of positive label and total count.
                                         This ratio can be used as balancing parameter in the model.
        """
        self.trainX = trainX
        self.trainy = trainy
        self.testX = testX
        self.testy = testy
        if class_weights is not None:
            self.class_weights = class_weights

    def train(self, nrounds=800, GridSearch=None, params=None, save_model=False, file_name=None,
              path=None, random_state=33):
        """
        The function trains XGBoost model with the specified parameter and saves the model in pickle format in the path
        Args:
            nrounds (int): epochs or nrounds for training
            GridSearch (Bool or None): If True, then performs random gridsearch on the parameters and trains the model
                                        on best parameter
            params (path or list):If gridsearch is implied then pass dict of parameters else for single training pass JSON
                         path consisting of training parameters.
            save_model (Bool): If True, the save the model after training is completed.
            file_name (str): file name for the model
            path (str): path to save the model after training
            random_state (int): seed for reproducbility

        Returns:
            model (): Trained XGBoost model
        """
        if hasattr(self, 'class_weights'):
            xgb_model = xgb.XGBClassifier(objective='binary:logistic', silent=True, n_estimators=nrounds,
                                          scale_pos_weight=self.class_weights, random_state=random_state)
        else:
            xgb_model = xgb.XGBClassifier(objective='binary:logistic', silent=True, n_estimators=nrounds,
                                          random_state=random_state)
        # TODO: Implemented only binary objective function, This has to be changed for multi-class
        if not GridSearch:
            dtrain = xgb.DMatrix(self.trainX, label=self.trainy)
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=nrounds
            )
            if save_model:
                file_name = str(file_name) + '.pkl'
                self.save_model(filename=file_name, path=path)
            return self.model
        else:
            kfold_5 = KFold(random_state=random_state, shuffle=True, n_splits=5)
            if multiprocessing.cpu_count() > 60:
                logger.info('CPUs core limited to 60')
                n_jobs = 60
            else:
                logger.info('All CPU cores are used')
                n_jobs = -1
            random_search = RandomizedSearchCV(xgb_model,
                                               param_distributions=params,
                                               cv=kfold_5,
                                               n_iter=5,
                                               scoring='recall',
                                               error_score=0,
                                               verbose=3,
                                               n_jobs=n_jobs)
            random_search.fit(self.trainX, self.trainy)
            best_params = random_search.best_params_
            best_params['objective'] = 'binary:logistic'
            if hasattr(self, 'class_weights'):
                best_params['scale_pos_weight'] = self.class_weights
            dtrain = xgb.DMatrix(self.trainX, label=self.trainy)
            self.model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=nrounds
            )
            if save_model:
                file_name = str(file_name) + '.pkl'
                self.save_model(filename=file_name, path=path)
                json_path = os.path.join(path, 'best_parameters_model.json')
                with open(json_path, 'w') as file:
                    json.dump(best_params, file)
                logger.info('Saved model and parameters in: {}'.format(path))
            return self.model

    def predict(self):
        """
        The function converts the data into "DMatrix" format and predicts
        Returns:
            self.model.predict(dtest): The probability for true positive
        """
        if self.testy is not None:
            dtest = xgb.DMatrix(self.testX, label=self.testy)
        else:
            dtest = xgb.DMatrix(self.testX)
        return self.model.predict(dtest)

    def save_model(self, filename, path):
        """
        The function saves the trained model to the location in the pickle format
        Args:
            filename (str): The file name to save the model in the pickle format
            path (str): The path to save the model
        """
        logger.info('Saving model as pickle file to the path: {}'.format(path))
        model_path = os.path.join(path, filename)
        pickle.dump(self.model, open(model_path, 'wb'))
        logger.info('Model saved')

    def load_model(self, filename, path):
        """
        The function loads model from location passed. The function only considered  model with ".pkl" extension
        Args:
            filename (str): the model file name to load
            path (str): the location where model is kept

        """
        logger.info('Loading Model from the location'.format(path))
        model_path = os.path.join(path, filename)
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            logger.info('Model loaded successfully')
        except Exception as error:
            logger.info('Unable to load model from the path: {}'.format(model_path) + repr(error))

    def feature_importance(self, path):
        """
        The function returns most important features/variables that is used in building XGBoost model
        The importance is determined by gain scores
        Returns:
            data (pd.DataFrame): The data frame consists of feature name and gain scores
        """
        feature_important = self.model.get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by='score', ascending=False)
        data = data.reset_index()
        data.to_csv(os.path.join(path, 'feature_importance.csv'), index=False)
        return data

    def shap_explanation(self, path, large_data: bool = False):
        if not large_data:
            data = self.trainX
        else:
            data = self.trainX.sample(frac=0.1)
        explainer = shap.TreeExplainer(self.model, data=data, feature_perturbation='interventional',
                                       model_output="probability")
        shap_values = explainer(data)
        create_and_save_shap_plots(shap_values=shap_values, dirName=path)


def create_and_save_shap_plots(shap_values, dirName):
    name = dirName.split("\\")[-1]
    logger.info(dirName)
    save_a_plot(shap_plot(plot=shap.plots.bar, shap_values=shap_values, title=name + "_shap_bar_plot", max_display=15),
                full_dirName=os.path.join(dirName, 'shap_bar_plot.png'))
    save_a_plot(shap_plot(plot=shap.plots.beeswarm, shap_values=shap_values, title=name + "_shap_beeswarm_plot"),
                full_dirName=os.path.join(dirName, 'shap_beeswarm_plot.png'))

    mean_abs_shap_values = np.abs(shap_values.values).mean(0)
    position_sorted_abs_values = np.argsort(mean_abs_shap_values).tolist()[::-1]
    top_x_features = [(shap_values.feature_names[i], mean_abs_shap_values[i]) for i in
                      position_sorted_abs_values[:10]]
    logger.info(top_x_features)


def shap_plot(plot, shap_values, title, **kwargs):
    plot(shap_values, show=False, **kwargs)
    plt.title(title)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def save_a_plot(plot, full_dirName):
    plot.savefig(full_dirName)
    logger.info("Saved {}".format(full_dirName.split("\\")[-1]))
    plt.close()
