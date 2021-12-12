"""
This file contains functions that help with evaluating performance of models
"""
from sklearn.metrics import mean_squared_error
from numpy import mean

def test():
    """Test function"""
    print("Imported Correctly")

def calculate_mse(pred, true):
    """
    calculates the mean squared error for a model.

    Parameters
    ----------
    pred : list 
        predicted feature values

    true : list 
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
