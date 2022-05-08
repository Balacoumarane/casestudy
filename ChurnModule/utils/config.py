from typing import Mapping, Optional, List, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .log import get_logger

logger = get_logger(__name__)


class MappingTableInfo(object):

    def __init__(self, mapping_sheet_path: Path = None):
        self.mapping_sheet_path = mapping_sheet_path

    def get_mapping_info(self) -> Union[Optional[List[Union[Mapping, list]]]]:
        """
        The function reads the excel sheet and returns the rename dict, list of columns to keep and datatypes.

        Returns:
            mapping_info (List): List contains rename dict, list of columns to keep and datatypes for data table

        """
        excel_path = self.mapping_sheet_path
        assert isinstance(excel_path, str)
        sheets_dict = pd.read_excel(excel_path, sheet_name=None)
        try:
            data_mapping_table = sheets_dict['data']
            data_col_list = list(data_mapping_table.loc[data_mapping_table['REQUIRED'] == 'YES', 'COLUMNS'].unique())
            # filter mapping table with only important column
            data_mapping_table = data_mapping_table[data_mapping_table['COLUMNS'].isin(data_col_list)]
            data_datatype_dict = create_dict(data_mapping_table, key_column='COLUMNS', value_column='DATA_TYPE')
            mapping_info = [data_col_list, data_datatype_dict]
        except Exception as error:
            mapping_info = None
            logger.info('There is issue data mapping info. Please check data sheet and error is:' + repr(error))
        return mapping_info


def create_dict(df: pd.DataFrame, key_column: str = None, value_column: str = None) -> Mapping:
    """
    The function creates dictionary from two columns in dataframe

    Args:
        df (pd.DataFrame): The Dataframe which contains two columns.
        key_column (str):  The Column to be stored as key in the dict.
        value_column (str): The column to be considered as value in the dict.

    Returns:
        output_dict (Dict): The output dictionary
    """
    output_dict = pd.Series(df[value_column].values, index=df[key_column]).to_dict()
    return output_dict


class StandardizeData(object):
    """
    The class converts data into standard format
    """

    def __init__(self):
        pass

    def standardise_table_format(self, df: pd.DataFrame, list_info: list = None,
                                 date_format: str = '%d/%m/%Y') -> pd.DataFrame:
        """
        The function preforms standardisation on the dataframe.

        Args:
            df (pd.DataFrame): Dataframe to standardise
            list_info (list): Tuple contains rename dict, list of columns to keep and datatypes.
            date_format (str): date in string format default is: '%d/%m/%Y'

        Returns:
            data (pd.DataFrame): Dataframe with new column names

        """
        data = df.copy()
        standardised_data = self.modify_data_type(df=data, data_type_dict=list_info[1],
                                                  date_format=date_format)
        standardised_data = self.keep_columns(df=standardised_data, cols_list=list_info[0])
        logger.info('The table has been standardised in MR Format')
        return standardised_data

    @staticmethod
    def modify_data_type(df: pd.DataFrame, data_type_dict=None, date_format: str = '%d/%m/%Y',
                         datetime_format: str = '%d/%m/%Y %H:%M:%S') -> pd.DataFrame:
        """
        The function will change the datatype according to the type specified in dictionary

        Args:
            df (pd.DataFrame): dataframe
            data_type_dict (Dict): Dictionary that consists of column name and datatype
            date_format (str): string for date format
            datetime_format (str): string for datetime format

        Returns:
            data (pd.DataFrame): The modified dataframe
        """
        logger.info('Fixing datatypes ...')
        data = df.copy()
        for columns, data_type in data_type_dict.items():
            try:
                if data_type == 'datetime':
                    if data[columns].dtype == 'datetime64[ns]':
                        logger.info('The column {} is already in timestamp'.format(columns))
                    else:
                        data[columns] = pd.to_datetime(data[columns], errors='coerce', format=datetime_format)
                        logger.info('The column {} converted to timestamp'.format(columns))
                elif data_type == 'date':
                    if data[columns].dtype == 'datetime64[ns]':
                        logger.info('The column {} is already in date format'.format(columns))
                    else:
                        data[columns] = pd.to_datetime(data[columns], errors='coerce', format=date_format)
                        logger.info('The column {} converted to date format'.format(columns))
                elif data_type == 'float' or data_type == 'int':
                    data[columns] = pd.to_numeric(data[columns], errors='coerce')
                    logger.info('The column {} converted to float format'.format(columns))
                else:
                    data[columns] = np.where(data[columns].isnull(), data[columns], data[columns].astype(data_type))
                    logger.info('The column {} converted to {}'.format(columns, data_type))
                logger.info('Fixed datatypes !!!')
            except Exception as error:
                logger.info('The column is not present in the data' + repr(error))
        return data

    @staticmethod
    def keep_columns(df: pd.DataFrame, cols_list: list = None) -> pd.DataFrame:
        """
        The function keeps only the columns mentioned in the list

        Args:
            df (pd.DataFrame): dataframe
            cols_list (list): list of column names

        Returns:
            data (pd.DataFrame): Data consists of columns mentioned in the list
        """
        logger.info('Dropping unwanted columns...')
        data = df.copy()
        data = data[np.intersect1d(data.columns, cols_list)]
        logger.info('Kept only relevant column in the data !!!')
        return data
