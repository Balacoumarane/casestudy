import click
from ChurnModule import train_execute, get_logger

logger = get_logger(__name__)


@click.command()
@click.option('--train_file_path', default=".//data//train.csv", type=str,
              help='path to load training file')
@click.option('--mapping_file_path', default='.//config//columns_mapping_info.xlsx', type=str,
              help='file location for column mapping info')
@click.option('--profile_report', default=True, type=bool,
              help='if true, generates data profiling report')
@click.option('--profile_report_path', default='.//reports', type=str,
              help='mention method to load data from database')
@click.option('--save_data', default=False, type=bool,
              help='if true saves the train data in csv format in report path')
@click.option('--model_output_path', default=".//model", type=str,
              help='file location for subset rules')
@click.option('--target_ratio', default=0.2, type=float,
              help='target[churn] ratio in the dataset. Default is 20%')
@click.option('--reduce_dim', default=True, type=bool,
              help='if true reduces cardinality in the columns before one-hot encoding')
@click.option('--grid_search', default=False, type=bool,
              help='if true reduces performs randomised grid search')
@click.option('--train_test_split', default=0.2, type=float, help='train-test ratio split')
@click.option('--model_threshold', default=0.5, type=float, help='threshold to consider true positive')
@click.option('--random_state', default=42, type=int, help='random state to replicate the results')
def train_main(train_file_path: str = None, mapping_file_path: str = None, profile_report: bool = False,
               profile_report_path: str = None, save_data: bool = False, reduce_dim: bool = False,
               model_output_path: str = None, target_ratio: float = 0.1, train_test_split: float = 0.2,
               grid_search: bool = True, model_threshold: float = 0.5, random_state: int = 42):
    train_execute(train_file_path=train_file_path, mapping_file_path=mapping_file_path, profile_report=profile_report,
                  profile_report_path=profile_report_path, save_data=save_data, reduce_dim=reduce_dim,
                  model_output_path=model_output_path, target_ratio=target_ratio, train_test_split=train_test_split,
                  grid_search=grid_search, model_threshold=model_threshold, random_state=random_state)


if __name__ == "__main__":
    train_main()
