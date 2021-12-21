"""
Contains miscellaneous functions that help with evaluating performance of models
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def test():
    """Test function"""
    print("Imported Correctly")

def map_plot(img:str, title:str, xlabel:str, ylabel:str, extents:list, lats:list, longs:list, labels:list):
    """
    Plot labelled points on a background image (possibly a map)

    Parameters
    ----------
    img : str
        location of background image
    title : str
        title of image
    xlabel : str
        label on the x-axis
    ylabel : str
        label on the y-axis
    extents : list
        extent of image (useful for maps). also used to maintein aspect ratio.
    lats : list
        List of latitude points.
    longs : list
        List of longitude points.
    labels : list
        List of labels for each point.

    Returns
    -----------
    Nothing - Generates a matplotlib figure.
    """
    im = plt.imread(img)
    plt.rc('font', size = 14)
    plt.figure(f'{title}',figsize=(10,12))
    plt.imshow(im, extent= extents)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


    
    # changing aspect ratio of figure to match image 
    # source: https://stackoverflow.com/questions/45685726/python-scatter-plot-over-background-image-for-data-verification?noredirect=1&lq=1
    aspect=im.shape[0]/float(im.shape[1])*((extents[1]-extents[0])/(extents[3]-extents[2]))
    plt.gca().set_aspect(aspect)

    for i in range(len(longs)):
        plt.scatter(longs[i], lats[i])
        plt.annotate(f'{labels[i]}',xy=(longs[i],lats[i]))


def plot_time_series_data(title:str, xlabel:str, ylabel:str, times:pd.Series, values:pd.Series):
    plt.figure(f'{title}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rc('font', size = 12)
    times = times
    values = values
    plt.show()
    return


    
def oneHot(X, y):
    enc = OneHotEncoder()
    enc.fit(X, y)
    return


def make_features(periods:int, dataframe:pd.DataFrame, col:str, name:str):
    X = dataframe[col]
    X = X.to_frame()
    X.rename({col:name},axis = 1, inplace=True)

    X = X.shift(periods=periods, fill_value = 0) # should change fill value
    X.reset_index(inplace = True)
    X.drop(columns=['index'], inplace=True)
    # X.drop(axis=0, )
    return X