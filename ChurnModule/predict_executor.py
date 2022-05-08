from typing import NoReturn
import datetime as dt
import os
import time
from .data import LoadData, StandaloneProcess
from .model import DataPrepModel, XGBoostModel
from .utils import get_logger, overlap_elements

logger = get_logger(__name__)


def predict_execute(predict_file_path: str = None,  mapping_file_path: str = None, model_dir_path: str = None,
                    reduce_dim: bool = False, profile_report: bool = False, profile_report_path: str = None,
                    result_path: str = None) -> NoReturn:
    """

    Args:
        predict_file_path:
        mapping_file_path:
        model_dir_path:
        profile_report:
        profile_report_path:
        result_path:
        reduce_dim:

    Returns:

    """
    start = time.time()
    # create model output folder
    if os.path.exists(result_path):
        logger.info("Output folder exists.")
    else:
        logger.info("Output folder does not exist, creating folder: {}".format(result_path))
        os.makedirs(result_path)
    result_dir = 'Score_' + dt.datetime.fromtimestamp(time.time()).strftime("%m%d-%H-%M-%S")
    result_dir = os.path.join(result_path, result_dir)
    os.makedirs(result_dir)

    # load data
    logger.info('Loading Prediction data')
    load_data = LoadData(data_path=predict_file_path)
    org_predict_table = load_data.load()

    # Standalone processing
    standalone_processing = StandaloneProcess(data=org_predict_table, mapping_file_path=mapping_file_path)
    standard_predict_table = standalone_processing.process(report=profile_report, report_path=profile_report_path,
                                                           report_file_name='Data Quality Report-Prediction')

    # Model prediction
    logger.info('Starting predictive model.....')
    predict_processing = DataPrepModel(pos_label=None, input_df=standard_predict_table, target_name=None,
                                       index_name='cust_id', encode_target=False)

    # encode the categorical data into dummy variable
    cat_features = ['gender', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age',
                    'vehicle_damage', 'policy_sales_channel']
    corrected_dummy_columns, _ = overlap_elements(cat_features, standard_predict_table.columns.tolist())
    # fix columns names
    if not reduce_dim:
        logger.info('Not reducing dimension before one-hot encoding')
    else:
        logger.info('Reducing dimension before one-hot encoding')
        column_reduce_path = os.path.join(model_dir_path, 'EncodedColumnNames.json')
        standard_predict_table = predict_processing.group_top_values_dict(input_dict_path=column_reduce_path,
                                                                          df=standard_predict_table)

    model_data = predict_processing.encoding_technique(df=standard_predict_table, encoding_method='onehot',
                                                       cols=corrected_dummy_columns)
    model_data = model_data.set_index('cust_id')
    encoded_column_path = os.path.join(model_dir_path, 'EncodedColumnNames.json')
    model_data = predict_processing.fix_encoded_column(test_encoded_data=model_data,
                                                       path=encoded_column_path,
                                                       predict=True)
    model_data = model_data.fillna(0)
    logger.info('Running prediction..')
    # load model
    XGB_Model = XGBoostModel(testX=model_data)
    XGB_Model.load_model(path=model_dir_path, filename='XGB_Model.pkl')
    # prediction
    standard_predict_table["propensity"] = XGB_Model.predict()
    standard_predict_table["propensity"] = standard_predict_table["propensity"].round(decimals=4)
    prediction_file = 'Customer_Churnscore' + dt.datetime.fromtimestamp(time.time()).strftime("%m%d-%H-%M-%S") + '.csv'
    prediction_file = os.path.join(result_dir, prediction_file)
    standard_predict_table[['cust_id', 'propensity']].to_csv(prediction_file, index=False)
    logger.info('Prediction results are saved in {}'.format(prediction_file))
    end = time.time()
    logger.info('Total time to train the data is {}'.format(end - start))
