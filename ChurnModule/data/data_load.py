from typing import Tuple
import time
import pandas as pd

from ..utils import get_logger, StandardizeData, data_profile_report, MappingTableInfo

logger = get_logger(__name__)


class LoadData(object):

    def __init__(self, data_path: str = None):
        """

        Args:
            data_path (str): path to training file
        """
        assert isinstance(data_path, str)
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        """

        Returns:
            train_data (pd.DataFrame): train data set
            test_data (pd.DataFrame): test data set

        """
        logger.info('Loading data from csv file')
        try:
            start = time.time()
            data = pd.read_csv(self.data_path)
            data.columns = data.columns.str.lower()
            end = time.time()
            logger.info('Loaded data. Time taken is {:.2f} secs'.format(end - start))
            return data
        except Exception as error:
            logger.info('Unable to load data and issue is ' + repr(error))


class StandaloneProcess(object):

    def __init__(self, data: pd.DataFrame = None, mapping_file_path: str = None):
        """

        Args:
            data (pd.DataFrame): dataset for training or prediction
            mapping_file_path (str): path to load column mapping excel

        """
        assert isinstance(data, pd.DataFrame)
        self.data = data
        self.mapping_table_path = mapping_file_path

    def process(self, report: bool = False, report_path: str = None, report_file_name: str = None) -> pd.DataFrame:
        """

        Args:
            report:
            report_path:

        Returns:
            train_data_formatted:
            test_data_formatted:
            report_file_name:

        """
        # get mapping info
        logger.info('Loading mapping table info')
        mapping_info = MappingTableInfo(mapping_sheet_path=self.mapping_table_path)
        # Get Mapping/Standardised info from the Excel
        data_mapping_info = mapping_info.get_mapping_info()
        logger.info('Converting into standard format')
        convert_format = StandardizeData()
        logger.info('Converting train table')
        data_formatted = convert_format.standardise_table_format(df=self.data, list_info=data_mapping_info)
        if report:
            logger.info('Creating file report in the path {}'.format(report_path))
            assert isinstance(report_path, str)
            logger.info('Creating  file report in the path')
            data_profile_report(df=data_formatted, filename=report_file_name, path=report_path)
        else:
            logger.info('Data profiling not selected')
        return data_formatted
