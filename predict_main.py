import click
from ChurnModule import predict_execute, get_logger

logger = get_logger(__name__)


@click.command()
@click.option('--predict_file_path', default=".//data//test.csv", type=str,
              help='path to load predict file')
@click.option('--mapping_file_path', default='.//config//columns_mapping_info.xlsx', type=str,
              help='file location for column mapping info')
@click.option('--model_dir_path', default=".//model//XGB_Model_0311-17-26-25", type=str,
              help='file location for model')
@click.option('--reduce_dim', default=False, type=bool,
              help='if true reduces cardinality in the columns before one-hot encoding')
@click.option('--profile_report', default=True, type=bool,
              help='if true, generates data profiling report for test dataset')
@click.option('--profile_report_path', default='.//reports', type=str,
              help='mention method to load data from database')
@click.option('--result_path', default='.//score', type=str,
              help='if true saves the train data in csv format in report path')
def predict_main(predict_file_path: str = None, mapping_file_path: str = None, model_dir_path: str = None,
                 reduce_dim: bool = False, profile_report: bool = False, profile_report_path: str = None,
                 result_path: str = None):
    predict_execute(predict_file_path=predict_file_path, mapping_file_path=mapping_file_path,
                    model_dir_path=model_dir_path, reduce_dim=reduce_dim, profile_report=profile_report,
                    profile_report_path=profile_report_path, result_path=result_path)


if __name__ == "__main__":
    predict_main()
