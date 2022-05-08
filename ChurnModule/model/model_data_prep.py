import pandas as pd
import numpy as np
import json
import os
import re
from sklearn.model_selection import train_test_split
from ..utils import get_logger
from ..utils import SamplingMethods
from ..utils import count_encoding, onehot_encoding

logger = get_logger(__name__)


class DataPrepModel:

    def __init__(self, input_df, pos_label=None, target_name=None, index_name=None, output_path=None,
                 encode_target=False):
        """

        Args:
            input_df:
            pos_label:
            target_name:
            index_name:
            encode_target:
            output_path:
        """

        self.input_df = input_df
        self.output_path = output_path
        self.target_name = target_name
        self.index_name = index_name
        self.encode_target = encode_target
        self.pos_label = pos_label

    @staticmethod
    def keep_cols(df, imp_cols):
        """
        The functions keeps specified columns and drop others

        Args:
            df (pd.DataFrame):  data
            imp_cols (list): list of columns

        Returns:
            data (pd.DataFrame): Data with specified column list

        """
        data = df.copy()
        data = data.loc[:, df.columns.isin(imp_cols)]
        return data

    @staticmethod
    def take_top_n(df, column_list, n: float = 0.03, path_save=None):
        """
        The function keeps the most occurring elements and bins others in order to reduce the
        cardinality of the categories

        Args:

            df (pd.dataframe): data
            column_list (list): list of columns to reduce the cardinality
            n (float): The threshold to consider whether to keep value or not (value= count/Total).
            path_save (str or None): path to save the column name and values considered in a JSON file.

        Raise:
            ValueError: if the column is not present in the list

        Return:
            data (pd.dataframe): modified data
            topn_var_dict (dict): Represents column name in key and values considered for each column in value

        """
        data = df.copy()
        topn_var_dict = {}
        for column_name in column_list:
            if column_name not in data.columns:
                raise ValueError("The column is not present")
            else:
                len_df = (data.shape[0])
                list_to_keep = data[column_name].value_counts().loc[
                    lambda x: x > round(n * len_df)].to_frame().reset_index()
                value_list = list(list_to_keep['index'])
                data[column_name] = np.where(data[column_name].isin(value_list), data[column_name], "OTHERS")
                topn_var_dict[column_name] = value_list
        if path_save is not None:
            json_path = os.path.join(path_save, 'variable_value_considered.json')
            with open(json_path, 'w') as file:
                json.dump(topn_var_dict, file)
        return data, topn_var_dict

    @staticmethod
    def group_top_values_dict(df, input_dict_path):
        """
        The function groups the categorical values in the column. Dict is passed to the function and key in dict
        contains column name and value in dict represents value to consider for the columns.
        Values other than the dictionary value is grouped as 'OTHERS' for the respective columns

        Args:
            df (pd.DataFrame): input data
            input_dict_path (str or None): path to load dict

        Returns:
            data (pd.DataFrame):  modified data

        """
        data = df.copy()
        with open(input_dict_path) as f:
            input_dict = json.load(f)
        logger.info("Total variables are :{}".format(len(input_dict)))
        for key, value in input_dict.items():
            try:
                if key in df.columns:
                    data[key] = np.where(data[key].isin(value), data[key], "OTHERS")
            except:
                logger.info("There is no column :{}".format(key))
        return data

    @staticmethod
    def replace_na_constant(df, column_dict):
        """
        The function replaces the NA in column by constant value

        Args:
            df (pd.DataFrame): input data
            column_dict (dict): The dict where key is column name and value is column value

        Returns:
            data (pd.DataFrame): Data with NAs in the columns are filled

        """
        data = df.copy()
        for col_name, value in column_dict.items():
            try:
                if col_name in df.columns:
                    data[col_name] = df[col_name].fillna(value)
            except Exception as error:
                logger.info("Column {} is not present in the dataframe and error is ".format(col_name) + repr(error))

        return data

    @staticmethod
    def replace_category(df, column_list):
        """
        The function replaces the NA in categorical column by taking mode (most frequent item)

        Args:
            df (pd.DataFrame): input data
            column_list (list): list of categorical columns

        Returns:
            data (pd.DataFrame): Data with NAs in the columns are filled

        """
        data = df.copy()
        for col_name in column_list:
            try:
                if col_name in df.columns:
                    data[col_name] = data[col_name].fillna(data[col_name].mode().iloc[0])
            except Exception as error:
                logger.info("Skipping Column {}. It is not present in the dataframe.".format(col_name) + repr(error))
        return data

    @staticmethod
    def replace_mean(df, column_list):
        """
        The function replaces the NA in numeric column by taking mean values

        Args:
            df (pd.DataFrame): input data
            column_list (list): list of numeric columns

        Returns:
            data (pd.DataFrame): Data with NAs in the columns are filled

        """
        data = df.copy()
        for col_name in column_list:
            try:
                if col_name in df.columns:
                    data[col_name] = data[col_name].fillna(data[col_name].mean())
            except Exception as error:
                logger.info("Skipping Column {}.It is not present in the dataframe.".format(col_name) + repr(error))
        return data

    def prepare_train_test(self, df, split_ratio, seed=33):
        """
        The function takes the dataframe as input and splits into test and train

        Args:
            df (pd.DataFrame): input data to split train and test
            split_ratio: The ratio for train and test split
            seed (int): random seed for reproducibility

        Returns:
            trainX (pd.DataFrame): training predictor data
            trainy (pd.DataFrame): training target variable
            testX (pd.DataFrame): test predictor data
            testy (pd.DataFrame): test target variable

        """
        data = df.copy()
        data = data.set_index(self.index_name)

        if self.encode_target:
            data[self.target_name] = np.where(data[self.target_name] == self.pos_label, 1, 0)

        X = data.drop([self.target_name], axis=1)
        y = data[self.target_name]
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=split_ratio, random_state=seed)
        logger.info("Total positive_label in training data is :{}".format(trainy.value_counts()[1]))
        logger.info("Total positive_label in test data is :{}".format(testy.value_counts()[1]))

        return trainX, testX, trainy, testy

    def convert_id_index(self, df):
        """
        The function set claim identifier as index and drop target variable
        Args:
            df (pd.DataFrame): input data

        Returns:
            data (pd.DataFrame): modified data

        """
        data = df.copy()
        data = data.set_index(self.index_name)
        data = data.drop([self.target_name], axis=1)
        return data

    @staticmethod
    def sample_train(trainX, trainy, sampling_method=None, random_state=56,
                     rus_sampling_ratio=0.5, smote_sampling_ratio=0.5,
                     smoterus_sampling_ratio=0.5, smote_k_neighbors=5,
                     categorical_features_index=None, smotenc_sampling_ratio=0.5,
                     smotenc_k_neighbors=5, smotesvm_sampling_ratio=0.5,
                     smotesvm_k_neighbors=5, smotesvm_m_neighbors=10, smotesvm_stepsize=0.5,
                     adasyn_sampling_ratio=0.25, adasyn_k_neighbors=5):
        """
        The function performs sampling on the training dataset

        Args:
            trainX (pd.DataFrame):
            trainy (pd.Series):
           sampling_method (str or None): Enter sampling method. Currently implemented ("RandomUnderSample",
                                            "SmoteUnderSample","SmoteNC", "SmoteSVM", "ADASYN")
           random_state (int): random seed for reproducibility
           rus_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority class
                                              over the number of samples in the majority class after resampling
           smote_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority
                                                class over the number of samples in the majority class after resampling.
           smoterus_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority class
                                                   over the number of samples in the majority class after resampling
           smote_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
           categorical_features_index (int or list): index for categorical features
           smotenc_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority class
                                                  over the number of samples in the majority class after resampling
           smotenc_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
           smotesvm_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority class
                                                  over the number of samples in the majority class after resampling
           smotesvm_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
           smotesvm_m_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.
           smotesvm_stepsize (int): Step size when extrapolating
           adasyn_sampling_ratio (int or float): It corresponds to the desired ratio of the number of samples in the minority class
                                                 over the number of samples in the majority class after resampling
           adasyn_k_neighbors (int):  number of nearest neighbours to used to construct synthetic samples.

        Returns:
            SampletrainX (pd.DataFrame):
            Sampletrainy (pd.series):

        """
        if sampling_method is not None:
            logger.info("Implementing sampling method: {}".format(sampling_method))
            sampling = SamplingMethods(trainX, trainy)
            SampletrainX, Sampletrainy = sampling.sampling_technique(random_state=random_state,
                                                                     sampling_method=sampling_method,
                                                                     rus_sampling_ratio=rus_sampling_ratio,
                                                                     smote_sampling_ratio=smote_sampling_ratio,
                                                                     smoterus_sampling_ratio=smoterus_sampling_ratio,
                                                                     smote_k_neighbors=smote_k_neighbors,
                                                                     categorical_features_index=categorical_features_index,
                                                                     smotenc_sampling_ratio=smotenc_sampling_ratio,
                                                                     smotenc_k_neighbors=smotenc_k_neighbors,
                                                                     smotesvm_sampling_ratio=smotesvm_sampling_ratio,
                                                                     smotesvm_k_neighbors=smotesvm_k_neighbors,
                                                                     smotesvm_m_neighbors=smotesvm_m_neighbors,
                                                                     smotesvm_stepsize=smotesvm_stepsize,
                                                                     adasyn_sampling_ratio=adasyn_sampling_ratio,
                                                                     adasyn_k_neighbors=adasyn_k_neighbors)
            return SampletrainX, Sampletrainy

    @staticmethod
    def encoding_technique(df, encoding_method=None, cols=None):
        """
        The function does takes the column and converts the categorical value in column to numeric value based on the
        Methods. currently only two methods are supported. 'count' and 'onehot' encodings are implemented"

        Args:
            df (pd.DataFrame): data
            encoding_method (str or None): encoding method ('count' or 'onehot')
            cols (list or None): list of categorical columns for encoding

        Returns:

        """
        data = df.copy()
        if encoding_method is None or encoding_method not in ['count', 'onehot']:
            raise NotImplementedError("Only 'count' and 'onehot' encodings are implemented")
        if encoding_method == "count" and cols is not None:
            data = count_encoding(data, cols_list=cols)
        if encoding_method == "onehot" and cols is not None:
            data = onehot_encoding(data, cols_list=cols)
        return data

    @staticmethod
    def fix_encoded_column(train_encoded_data=None, test_encoded_data=None, path=None, predict=False):
        """
        The function aligns the training and testing columns which is important as feature names should align.
        While training it saves trained column headers in the json file and in prediction it loads the json containing
        training feature names and aligns prediction data to have same column name as training data

        Args:
            train_encoded_data (pd.DataFrame): training data
            test_encoded_data (pd.DataFrame): test/prediction data
            path (str or None): path to load the Json file containing trained feature names
            predict (Bool):if true, set function for prediction mode

        Returns:
            train_encoded_data (pd.DataFrame): aligned training data
            test_encoded_data (pd.DataFrame): aligned test/prediction data

        """
        regex = re.compile(r"[\[\]<]", re.IGNORECASE)
        if not predict:
            logger.info("Aligning training columns and test columns...")
            fix_encoded_columns_dict = {}
            train_encoded_data.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col
                                          for col in train_encoded_data.columns.values]
            # train_encoded_data.columns = train_encoded_data.columns.str.replace(' ', '_')
            fix_encoded_columns_dict['ENCODED_TRAINING_COLS_NAMES'] = list(train_encoded_data.columns)
            if path is not None:
                json_path = os.path.join(path, 'EncodedColumnNames.json')
                with open(json_path, 'w') as file:
                    json.dump(fix_encoded_columns_dict, file)
                logger.info("Saving training column names in the path: {}".format(path))
            test_encoded_data.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col
                                         for col in test_encoded_data.columns.values]
            # test_encoded_data.columns = test_encoded_data.columns.str.replace(' ', '_')
            missing_cols = set(train_encoded_data.columns) - set(test_encoded_data.columns)
            for c in missing_cols:
                test_encoded_data[c] = 0
            test_encoded_data = test_encoded_data[train_encoded_data.columns]
            return train_encoded_data, test_encoded_data
        else:
            if path is not None:
                logger.info("Aligning prediction columns similar to training columns...")
                with open(path) as f:
                    train_encoded_data = json.load(f)
                train_columns = train_encoded_data['ENCODED_TRAINING_COLS_NAMES']
                test_encoded_data.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col
                                             for col in test_encoded_data.columns.values]
                # test_encoded_data.columns = test_encoded_data.columns.str.replace(' ', '_')
                missing_cols = set(train_columns) - set(test_encoded_data.columns)
                for c in missing_cols:
                    test_encoded_data[c] = 0
                test_encoded_data = test_encoded_data[train_columns]
                return test_encoded_data
