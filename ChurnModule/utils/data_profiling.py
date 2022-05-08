from typing import NoReturn
import pandas as pd
from pathlib import Path
import os
from pandas_profiling import ProfileReport
from .log import get_logger

logger = get_logger(__name__)


def data_profile_report(df: pd.DataFrame, filename: str = "Data Profiling Report", path: str = None) -> NoReturn:
    """
    The function creates profiling report on the table and saves the report in HTML format.

    Args:
        df (pd.DataFrame): Input data
        filename (str): filename for the HTML report
        path (Path or str): Path to save the report.

    Returns:
        NoReturns
    """
    data_report = ProfileReport(df=df,
                                title=filename,
                                sort='ascending',
                                minimal=True,
                                progress_bar=True,
                                samples=None,
                                correlations={
                                    "pearson": {"calculate": True},
                                    "spearman": {"calculate": True},
                                    "kendall": {"calculate": False},
                                    "phi_k": {"calculate": False},
                                    "cramers": {"calculate": False},
                                },
                                duplicates=None,
                                interactions={'continuous': False},
                                missing_diagrams={
                                    'bar': True,
                                    'matrix': False,
                                    'heatmap': False,
                                    'dendrogram': False,
                                },
                                explorative=True,
                                html={'style': {'theme': 'flatly'}}
                                )

    # Output the report to HTML
    if path is not None:
        assert isinstance(path, str)
        report_name = filename + '.html'
        path_save = os.path.join(path, report_name)
        data_report.to_file(path_save)
        logger.info("The file {} is saved in {}".format(report_name, path))
    else:
        logger.info("Path is not provided. Can't save the report")
