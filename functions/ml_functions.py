import numpy as np

def generate_predictions(model, test_data):
    """
    Generates predictions on a dataset using an array of models 
    Parameters
    ----------
    model : linear_model.Ridge
        features
    test_data : numpy.ndarray
        input features
    Returns
    ----------
    predictions_array : array
    """
    predictions_array = np.ndarray(len(test_data), dtype= np.float64 )
    
    predictions_array = model.predict(test_data)

    return predictions_array