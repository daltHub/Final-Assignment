"""
This file contains functions that help with evaluating performance of models
"""
from numpy.lib.function_base import average
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from numpy import mean

def test():
    """Test function"""
    print("Imported Correctly")

def calculate_mse(pred, true):
    """
    calculates the mean squared error for a model.

    Parameters
    ----------
    pred : array 
        predicted feature values

    true : array 
        true feature values

    Returns
    ----------
    average_loss : float
        mean loss between predictions and truth

    """

    losses = mean_squared_error(y_true=true, y_pred=pred)
    average_loss = mean(losses)
    # print(f"average loss = {average_loss}")
    return average_loss
