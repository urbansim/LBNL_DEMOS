import numpy as np
import pandas as pd
from scipy.special import softmax

def simulation_mnl(data, coeffs):
    """
    Simulate choices using a Multinomial Logit (MNL) model.

    This function calculates the utility of each choice alternative using the 
    provided data and coefficients, applies the softmax function to compute 
    choice probabilities, and simulates choices based on these probabilities.

    Args:
        data (pd.DataFrame or np.ndarray): The input data containing features 
                                           for each choice alternative.
        coeffs (np.ndarray): Coefficients for the MNL model, used to calculate 
                             the utility of each choice alternative.

    Returns:
        pd.Series: A Pandas Series containing the simulated choices, indexed 
                   by the input data's index.
    """
    utils = np.dot(data, coeffs)
    base_util = np.zeros(utils.shape[0])
    utils = np.column_stack((base_util, utils))
    probabilities = softmax(utils, axis=1)
    s = probabilities.cumsum(axis=1)
    r = np.random.rand(probabilities.shape[0]).reshape((-1, 1))
    choices = (s < r).sum(axis=1)
    return pd.Series(index=data.index, data=choices)

def calibrate_model(model, target_count, threshold=0.05):
    """
    Calibrate a model to match a target count.

    This function adjusts the model's parameters to ensure that the predicted 
    outcomes closely match the target count. It iteratively updates the model's 
    parameters until the error between the predicted and target counts is within 
    a specified threshold.

    Args:
        model: The model object to be calibrated. It must have `run` and 
               `fitted_parameters` attributes.
        target_count (int or float): The target count that the model's predictions 
                                     should match.
        threshold (float, optional): The acceptable error threshold for calibration. 
                                     Defaults to 0.05.

    Returns:
        np.ndarray: The calibrated predictions from the model.
    """
    model.run()
    predictions = model.choices.astype(int)
    predicted_share = predictions.sum() / predictions.shape[0]
    target_share = target_count / predictions.shape[0]

    error = (predictions.sum() - target_count.sum())/target_count.sum()
    while np.abs(error) >= threshold:
        model.fitted_parameters[0] += np.log(target_count.sum()/predictions.sum())
        model.run()
        predictions = model.choices.astype(int)
        predicted_share = predictions.sum() / predictions.shape[0]
        error = (predictions.sum() - target_count.sum())/target_count.sum()
    return predictions