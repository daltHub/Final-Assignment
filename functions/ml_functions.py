import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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


def train_Kfold_ridge(X_features, y_features, alpha_value):
    """
    Uses K-fold cross validation 
    Parameters
    ----------
    X_features : array 
        features
    y_features : array
        target features
    alpha_value : float
        parameter for training
    Returns
    ----------
    mean error : float
    standard error : float
    """
    kf = KFold(n_splits=5)
    model = Ridge(alpha=alpha_value)
    errs = []
    # model = linear_model.Lasso(alpha=1/(2*c_value), max_iter=1000000000).fit()
    for train, test in kf.split(X_features):
        model.fit(X_features[train],y_features[train])
        ypred = model.predict(X_features[test])
        from sklearn.metrics import mean_squared_error
        # print("square error %f"%(mean_squared_error(y_features[test],ypred)))
        errs.append(mean_squared_error(y_features[test],ypred))
    # print(np.mean(errs))
    return np.mean(errs), np.std(errs)


def Kfold_for_alpha_ridge(X_features, y_features, alpha_range, title:str):
    """
    Uses K-fold cross validation with varied values of C
    Parameters
    ----------
    X_features : array 
        features
    y_features : array
        target features
    C_range : array of float
        parameters for training
    Returns
    ----------
    nothing
    """
    error_array = np.zeros(len(alpha_range))
    std_dev_array = np.zeros(len(alpha_range))
    for i in range(len(alpha_range)):
        # print("\n\n C = %f"%(C_range[i]))
        error_array[i], std_dev_array[i] = train_Kfold_ridge(X_features, y_features, alpha_range[i])


    # print(error_array)

    plt.figure(title)
    plt.errorbar(alpha_range, error_array, yerr=std_dev_array)
    plt.xlabel('Alpha value')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    # x = np.arange(len(error_array))
    # plt.bar(x, C_range, error_array)