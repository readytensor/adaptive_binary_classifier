from typing import List

import numpy as np
import pandas as pd


def create_predictions_dataframe(
    predictions_arr: np.ndarray,
    class_names: List[str],
    prediction_field_name: str,
    ids: pd.Series,
    id_field_name: str,
    return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df
