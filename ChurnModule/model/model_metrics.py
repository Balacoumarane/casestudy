from typing import Union
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, matthews_corrcoef, jaccard_score, f1_score, average_precision_score, \
    balanced_accuracy_score, brier_score_loss, log_loss, accuracy_score, \
    precision_score, recall_score, confusion_matrix

from ..utils import get_logger
logger = get_logger(__name__)


class HelperMetricCalculation:

    def __init__(self, y_true, y_pred, y_pred_proba, prob_threshold: float = None):
        self.y_true = y_true
        if prob_threshold is None:
            logger.info('Using 0.5 threshold to classify positive class')
            self.y_pred = y_pred
        else:
            logger.info('Using {} prob_threshold to classify positive class'.format(prob_threshold))
            self.y_pred = np.where(y_pred_proba >= prob_threshold, 1, 0)
        self.y_pred_proba = y_pred_proba

    def score(self, fun, top_x=None, **kwargs):
        """

        Args:
            fun:
            top_x:
            **kwargs:

        Returns:

        """
        check = exceptions(fun, self.y_true, self.y_pred)
        if check is not None:
            return check

        if top_x is not None:
            if self.y_pred_proba is not None and sum(self.y_pred_proba) != 0:
                df = pd.DataFrame({'y_true': self.y_true, 'y_pred_proba': self.y_pred_proba})
                df = df.sort_values(by=['y_pred_proba'], ascending=False)
                df_length = df.shape[0]
            else:
                return "_"
            if 0 < top_x < 1:
                num_top_p = int(df_length * top_x)
                y_pred = np.repeat((1, 0), (num_top_p, df_length - num_top_p))
            elif isinstance(top_x, int) and top_x > 0 and df_length - top_x > 0:
                y_pred = np.repeat((1, 0), (top_x, df_length - top_x))
            else:
                logger.info("top_x has to be between 0 and 1 or a positive integer or smaller than the data length")
                return "_"
            try:
                return fun(df.y_true, y_pred, **kwargs)
            except:
                return "_"
        return fun(self.y_true, self.y_pred, **kwargs)

    def score_proba(self, fun, **kwargs):
        """

        Args:
            fun:
            **kwargs:

        Returns:

        """
        check = exceptions(fun, self.y_true, self.y_pred_proba)
        if check is not None:
            return check
        return fun(self.y_true, self.y_pred_proba, **kwargs)


def metric_calculation(y, y_pred, y_pred_proba, prob_threshold: float = None) -> Union[dict, None]:
    """

    Args:
        y:
        y_pred:
        y_pred_proba:
        prob_threshold:

    Returns:

    """
    if y is None:
        logger.info('No true target variable in train to calc_metrics')
        return None

    s = HelperMetricCalculation(y, y_pred, y_pred_proba, prob_threshold=prob_threshold)

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    if len(class_counts) > 1:
        positive_class_share = f"{class_counts[1]}/{class_counts[0] + class_counts[1]} = " \
                               f"{round(class_counts[1] / (class_counts[0] + class_counts[1]), 3)}"
    else:
        positive_class_share = "Only one class: {}".format(class_counts.keys)

    tn, fp, fn, tp = s.score(confusion_matrix).ravel()
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    scores = {
        "positive_class_share": positive_class_share,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc_labels": s.score(roc_auc_score),
        "roc_auc_proba": s.score_proba(roc_auc_score),
        "accuracy": s.score(accuracy_score),
        "recall": s.score(recall_score),
        "recall_top_1%": s.score(recall_score, top_x=0.01),
        "recall_top_2%": s.score(recall_score, top_x=0.02),
        "recall_top_5%": s.score(recall_score, top_x=0.05),
        "recall_top_10%": s.score(recall_score, top_x=0.1),
        "precision": s.score(precision_score),
        "precision_top_1%": s.score(precision_score, top_x=0.01),
        "precision_top_2%": s.score(precision_score, top_x=0.02),
        "precision_top_5%": s.score(precision_score, top_x=0.05),
        "precision_top_10%": s.score(precision_score, top_x=0.1),
        "precision_top_100": s.score(precision_score, top_x=100),
        "precision_top_200": s.score(precision_score, top_x=200),
        "f1": s.score(f1_score),
        "f1_top_1%": s.score(f1_score, top_x=0.01),
        "f1_top_2%": s.score(f1_score, top_x=0.02),
        "f1_top_5%": s.score(f1_score, top_x=0.05),
        "f1_top_10%": s.score(f1_score, top_x=0.1),
        "mcc": s.score(matthews_corrcoef),
        "jaccard": s.score(jaccard_score),
        "average_precision": s.score(average_precision_score),
        "balanced_accuracy": s.score(balanced_accuracy_score),
        "brier": s.score(brier_score_loss),
        "log_loss": s.score(log_loss),
        "log_loss_top_1%": s.score(log_loss, top_x=0.01),
        "log_loss_top_5%": s.score(log_loss, top_x=0.05)
    }

    logger.info("The prediction resulted in the following scores: ")
    logger.info("{}".format([key + ':' + str(item) + ',' for key, item in scores.items()]))
    return scores


def exceptions(fun, a, b, c=[1]):
    if a is None or b is None or c is None or sum(a) == 0 or sum(b) == 0 or sum(c) == 0:
        if a is None or sum(a) == 0:
            logger.info("No calc_metrics calculated for " + str(fun.__name__) + " because true vector is None or contains only 0s")
        else:
            logger.info("No calc_metrics calculated for " + str(fun.__name__) + " because prediction is None or contains only 0s")
        if str(fun.__name__) == "confusion_matrix":
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])
        else:
            return np.nan
    else:
        return None