import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.ticker as mticker
import seaborn as sns
plt.style.use('seaborn')


def create_heatmap(df, size=(15,8), save_name=None):
    """
    Produces a correlation heat map from a panda dataframe and saves it if save_name input

    Arg:
        df(pdDataFrame): a dataframe to plot
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays heat map plot
        Stores graph in image folder if save_name is input
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        data=corr,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        ax=ax,
        annot=True,
        cbar_kws={"label": "Correlation",
                  "orientation": "horizontal", "pad": .2, "extend": "both"}
    )
    ax.set_title(
        "Heatmap of Correlation Between Attributes (Including Target)")
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def linearity_graph(model, X_test, y_test, size=(15,8), save_name=None):
    """
    Produces a linearity test scatter plot from a panda dataframe and saves it if save_name input

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the linearity plot
        Stores graph in image folder if save_name is input
    """

    preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=size)
    perfect_line = np.arange(y_test.min(), y_test.max())
    ax.plot(perfect_line, perfect_line, linestyle="--",
            color="orange", label="Perfect Fit")
    ax.scatter(y_test, preds, alpha=0.5)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def normality_graph(model, X_test, y_test, size=(15,8), save_name=None):
    """
    Produces a Q-Q plot from a panda dataframe and saves it if save_name input

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the normality plot
        Stores graph in image folder if save_name is input
    """
    fig, ax = plt.subplots(figsize=size)
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True, ax=ax)
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def homoscedasticity_graph(model, X_test, y_test, size=(15,8), save_name=None):
    """
    Produces a homoscedasticity test scatter plot from a panda dataframe and saves it if save_name input

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the homoscedasticity plot
        Stores graph in image folder if save_name is input
    """
    fig, ax = plt.subplots(figsize=size)
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    ax.scatter(preds, residuals, alpha=0.5)
    ax.plot(preds, [0 for i in range(len(X_test))],
            linestyle="--", color='orange', label="Perfect Fit")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual - Predicted Value")
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()


def one_hot_coef_graph(coef_df, categories, dropped_var, target_name='Price', increase_type='Percent', size=(15,8), rotate=None, fix_names = True, save_name=None):
    """
    Produces a bar graph of the coefficients from a panda dataframe and saves it if save_name input

    Arg:
        coef_df(pdDataFrame): a dataframe containing the coefficient to graph
        categories(array): a list of the category names whose coefficients you wish to graph
        dropped_var(str): the name of the variable droped during one-hot encoding
        target_name(str): the name of the target variable
        increase_type(str): the type of increase represented by the coefficient, either 'Percent' or 'Absolute'
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the bar graph of the listed categories coefficients
        Stores graph in image folder if save_name is input
    """
    coefs = []
    for cat in categories:
        coefs.append(float(coef_df[cat]))
    fig, ax = plt.subplots(figsize=size)
    x_labels = categories.copy()
    if fix_names:
        for i, cat in enumerate(categories):
            x_labels[i] = cat.replace('_', ' ').title()
    ax = plt.bar(x_labels, coefs)
    plt.ylabel("{x} Increase in {y}".format(x=increase_type, y=target_name))
    plt.title("Predicted {x} Increase in {y} Relative to {z}".format(
        x=increase_type, y=target_name, z=dropped_var))
    if rotate:
        plt.xticks(rotation = rotate)
    if save_name:
        plt.savefig(f'images/{save_name}.png')
    return plt.show()
