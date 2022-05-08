from typing import NoReturn
import datetime as dt
import os
import time
from .data import LoadData, StandaloneProcess
from .model import DataPrepModel, XGBoostModel, metric_calculation
from .utils import get_logger, overlap_elements

logger = get_logger(__name__)


def train_execute(train_file_path: str = None, mapping_file_path: str = None, profile_report: bool = False,
                  profile_report_path: str = None, save_data: bool = False, reduce_dim: bool = False,
                  model_output_path: str = None, target_ratio: float = 0.2, grid_search: bool = True,
                  train_test_split: float = 0.2, model_threshold: float = 0.5, random_state: int = 42) -> NoReturn:
    """

    Args:
        train_file_path:
        mapping_file_path:
        profile_report:
        profile_report_path:
        model_output_path:
        target_ratio:
        reduce_dim:
        grid_search:
        train_test_split:
        model_threshold:
        random_state:
        save_data:

    Returns:

    """
    start = time.time()
    # create model output folder
    if os.path.exists(model_output_path):
        logger.info("Output folder exists.")
    else:
        logger.info("Output folder does not exist, creating folder: {}".format(model_output_path))
        os.makedirs(model_output_path)
    model_dir = 'XGB_Model_' + dt.datetime.fromtimestamp(time.time()).strftime("%m%d-%H-%M-%S")
    model_dir = os.path.join(model_output_path, model_dir)
    os.makedirs(model_dir)

    # create file report path
    if os.path.exists(profile_report_path):
        logger.info("Output folder exists.")
    else:
        logger.info("Output folder does not exist, creating folder: {}".format(profile_report_path))
        os.makedirs(profile_report_path)

    # load data
    load_data = LoadData(data_path=train_file_path)
    org_train_table = load_data.load()
    # cleaning
    standalone_processing = StandaloneProcess(data=org_train_table, mapping_file_path=mapping_file_path)
    standard_train_table = standalone_processing.process(report=profile_report, report_path=profile_report_path,
                                                         report_file_name='Data Quality Report-Train')
    if profile_report_path is not None and save_data:
        logger.info('Saving cleaned file...')
        assert isinstance(profile_report_path, str)
        path_save = os.path.join(profile_report_path, 'train.csv')
        standard_train_table.to_csv(path_save)
        logger.info("The Cleaned file is saved in {}".format(profile_report_path))

    # Prep data for model
    logger.info('Training model')
    # Model training
    train_processing = DataPrepModel(pos_label=1, input_df=standard_train_table, target_name='response',
                                     index_name='cust_id', encode_target=False)
    # keep only important columns
    imp_features = ['cust_id', 'gender', 'age', 'driving_license', 'region_code',
                    'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
                    'policy_sales_channel', 'days_since_insured', 'response']

    processing_data = train_processing.keep_cols(df=standard_train_table, imp_cols=imp_features)
    # reduce frequency
    if not reduce_dim:
        logger.info('Not reducing dimension before one-hot encoding')
    else:
        logger.info('Reducing dimension before one-hot encoding')
        cols_reduce = ['policy_sales_channel', 'region_code']

        processing_data, var_list = train_processing.take_top_n(df=processing_data, column_list=cols_reduce, n=0.01,
                                                                path_save=model_dir)

    # Fill na in categorical column by taking the most frequent one/mode
    cat_features = ['gender', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age',
                    'vehicle_damage', 'policy_sales_channel']
    processing_data = train_processing.replace_category(df=processing_data, column_list=cat_features)

    # Fill na in categorical column by taking mean
    num_features = ['age', 'annual_premium', 'days_since_insured', 'response']
    processing_data = train_processing.replace_mean(df=processing_data, column_list=num_features)

    # Split train and test
    trainX, testX, trainy, testy = train_processing.prepare_train_test(df=processing_data, split_ratio=train_test_split)

    # Sample
    RusTrainX, RusTrainy = train_processing.sample_train(trainX, trainy, sampling_method="RandomUnderSample",
                                                         random_state=random_state, rus_sampling_ratio=target_ratio)
    # Encode the categorical data into dummy variable
    corrected_dummy_columns, _ = overlap_elements(cat_features, RusTrainX.columns.tolist())

    RusTrainX_encode = train_processing.encoding_technique(df=RusTrainX, encoding_method='onehot',
                                                           cols=corrected_dummy_columns)
    testX_encode = train_processing.encoding_technique(df=testX, encoding_method='onehot',
                                                       cols=corrected_dummy_columns)

    RusTrainX_encode, testX_encode = train_processing.fix_encoded_column(RusTrainX_encode, testX_encode,
                                                                         path=model_dir)

    # Define and train model
    XGB_Model = XGBoostModel(trainX=RusTrainX_encode, trainy=RusTrainy, testX=testX_encode, testy=testy)
    if not grid_search:
        logger.info('Initialising params for training')
        hyper_params = {
            "subsample": 0.8,
            "min_child_weight": 1,
            "max_depth": 5,
            "learning_rate": 0.1,
            "gamma": 2,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic"}
    else:
        logger.info('Initialising params for grid search')
        hyper_params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.03, 0.1]
        }
    model_filename = 'XGB_Model'
    XGB_Model.train(nrounds=800, GridSearch=grid_search, params=hyper_params, save_model=True, file_name=model_filename,
                    path=model_dir, random_state=random_state)
    test_y_pred_proba = XGB_Model.predict()
    logger.info("Score Test: ")
    _ = metric_calculation(testy, _, test_y_pred_proba, prob_threshold=model_threshold)
    # feature analysis
    XGB_Model.feature_importance(path=model_dir)
    XGB_Model.shap_explanation(path=model_dir, large_data=True)
    end = time.time()
    logger.info('Total time to train the data is {}'.format(end - start))
