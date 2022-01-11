import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, ShuffleSplit


def missing_indicator(df, column):
    """
    Produces an array of booleans representing missing values from column

    Arg:
        df(pdDataFrame): A dataframe with a column to create a missing indicator array from
        column(str): A string which is the column label of the desired column

    Return:
        missing(array): A numpy array containing booleans coresponding to the null values of the column
    """
    c = df[[column]]
    miss = MissingIndicator()
    miss.fit(c)
    missing = miss.transform(c)
    return missing


def impute_values(df, column, type='median'):
    """
    Produces an array of booleans representing missing values from column

    Arg:
        df(pdDataFrame): A dataframe with a column to impute missing values to
        column(str): A string which is the column label of the desired column
        type(str): A name of a method to use to impute values from the sklearn SimpleImputer

    Return:
        imputed(array): A numpy array containing imputed values for the missing data
    """
    c = df[[column]]
    imputer = SimpleImputer(strategy=type)
    imputer.fit(c)
    imputed = imputer.transform(c)
    return imputed


def oridinal_encode(df, column):
    """
    Produces an array of category names and an array of integers representing the categories

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        column(str): A string which is the column label of the desired column

    Return:
        cat(array): A numpy array containing strings of the categories in the column
        encoded(array): A numpy array containing intergers representing the categories
    """
    c = df[[column]]
    encoder = OrdinalEncoder()
    encoder.fit(c)
    cat = encoder.categories_[0]
    encoded = encoder.transform(c)
    encoded = encoded.flatten()
    return cat, encoded


def one_hot_encode(df, column, prefix=None):
    """
    Produces an array of category names and an array of array contain boolean representing the categories

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        column(str): A string which is the column label of the desired column

    Return:
        cats(array): An array containing strings of the categories in the column (only returned if prefix not given)
        prefixed(array): An array containing strings of the categories in the column with the specified prefix (only returned if prefix given)
        encoded(array): A numpy array containing arrays containing booleans representing the categories
    """
    c = df[[column]]
    ohe = OneHotEncoder(categories="auto", sparse=False,
                        handle_unknown="ignore")
    ohe.fit(c)
    cats = ohe.categories_
    cats = list(cats[0])
    encoded = ohe.transform(c)
    if prefix:
        prefixed = []
        for cat in cats:
            prefixed.append(prefix+cat)
        return prefixed, encoded
    return cats, encoded


def collinearity_check(df, min=0.75, max=1, exclude_max=True):
    """
    Produces a dataframe listing all the correlations between the variables in the dataframe
    in between the the input min and max values

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        min(float64): A float between 0 and 1
        max(float64): A float between 0 and 1, usually 1

    Return:
        cat(array): A numpy array containing strings of the categories in the column
        encoded(array): A numpy array containing arrays containing booleans representing the categories
    """
    check = df.corr().abs().stack().reset_index().sort_values(0, ascending=False)
    check['pairs'] = list(zip(check.level_0, check.level_1))
    check.set_index(['pairs'], inplace=True)
    check.drop(columns=['level_1', 'level_0'], inplace=True)
    check.columns = ['cc']
    if exclude_max:
        return check[(check.cc >= min) & (check.cc < max)]
    return check[(check.cc >= min) & (check.cc <= max)]


def mean_sq_error(model, X_train, y_train, X_test, y_test, scaler=None):
    """
    Gives the mean square error in the original units using the scikit scaler used to scale the data.

    Arg:
        model: Linear regression model produced from sklearn LinearRegression
        X_train(pdDataFrame): A dataframe containing all the predictor variables the model was trained on
        y_train(pdSeries): A series containing the target varible the model was trained on
        X_test(pdDataFrame): A dataframe containing all the predictor variables the model will be tested against
        y_test(pdSeries): A series containing the target varible the model will be tested against
        scaler: either a string identifying the data was log scaled or a scikit scaler used to scale the target with the target column as the first value

    Return:
        Prints both training and test mse
        train_mse(float64): A float representing the mean square error of the model relative to the training data
        test_mse(float64): A float representing the mean square error of the model relative to the testing data
    """

    if scaler == 'logp1':
        y_hat_train = np.expm1(model.predict(X_train))
        y_hat_test = np.expm1(model.predict(X_test))
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
    elif scaler:
        y_hat_train = model.predict(X_train)*scaler.scale_[0]+scaler.mean_[0]
        y_hat_test = model.predict(X_test)*scaler.scale_[0]+scaler.mean_[0]
        y_train = y_train*scaler.scale_[0]+scaler.mean_[0]
        y_test = y_test*scaler.scale_[0]+scaler.mean_[0]
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
    else:
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)

    print('Train Mean Squared Error:', train_mse)
    print('Test Mean Squared Error:', test_mse)
    return train_mse, test_mse


def cross_val(model, X_train, y_train, splits=5, test_size=0.25, random_state=0):
    """
    Gives the mean of the R-squared values for the model for the specified number of Kfolds

    Arg:
        model: Linear regression model produced from sklearn LinearRegression
        predictors(pdDataFrame): A dataframe containing all the predictor variables from the model
        target(pdSeries): A series containing the target varible of the model
        fold(int64): The number of Kfolds to perform in the cross validation

    Return:
        Prints mean of the training and test R-squared scores
        scores(dict): A dictionary containing arrays for the following keys: fit_time, score_time, test_score, train_score
    """
    splitter = ShuffleSplit(
        n_splits=splits, test_size=test_size, random_state=random_state)
    scores = cross_validate(estimator=model, X=X_train,
                            y=y_train, return_train_score=True, cv=splitter)
    print("Train score:     ", scores["train_score"].mean())
    print("Validation score:", scores["test_score"].mean())
    return scores


def predict_median_effect(df, variable_column, model, target='price'):
    """
    Produce a dataframe containing predictions for the target variable from the model for the median values
    in the input variable column.

    Arg:
        df(pdDataFrame): a dataframe containing the training data of the model
        variable_column(str): a label of a variable column in the training data
        model: a scikit linear regression model produced from the training data
        target: the name of the target variable the model predicts the values of

    Return:
        predict_df(pdDataFrame): a DataFrame containing predictions for the target variable from the model for the median values
        in the input variable column
    """
    predict_df = df.groupby(variable_column).median().reset_index()
    predict_df.insert(0, target, model.predict(predict_df))
    return predict_df


def give_prediction(model, variables, target='price', rnd=2, scaler=None, scaled_columns=[]):
    """
    Gives prediction about the target variable in unscaled values from unscaled variable inputs

    Arg:
        model: a scikit linear regression model
        variables(list): a list of the variable columns the model was run on in original order
        target(str): name of the target varible that is predicted

    Return:
        prediction(float): a float representing the predicted value
    """
    if scaler:
        z = list(zip(scaler.mean_, scaler.scale_))
        scale_dict = dict(zip(scaled_columns, z))
    values = []
    for variable in variables:
        print('Input {}:'.format(variable))
        raw_value = float(input())
        if variable in scaled_columns:
            value = (raw_value-scale_dict[variable][0])/scale_dict[variable][1]
        else:
            value = raw_value
        values.append(value)
    array_values = np.array(values)
    prediction = np.sum(array_values * model.coef_) + model.intercept_
    if target in scaled_columns:
        prediction = prediction*scale_dict[target][1] + scale_dict[target][0]
    return round(prediction, ndigits=round)


def king_county_prediction(model, variables, target='price', rnd=2, scaler=None, scaled_columns=[]):
    var_dict = dict.fromkeys(variables, 0)
    print('Input Number of Bathrooms:')
    num_bath = input()
    if (round(float(num_bath)) == float(num_bath)) and num_bath !='1':
        num_bath = num_bath +'.0'
    elif num_bath !='1':
        var_dict['bathrooms_'+num_bath]
    print('Input Number of Bedrooms:')
    num_bed = input()
    if num_bed != '1':
        var_dict['bedrooms_'+num_bed]
    print('Input Number of Floors:')
    num_floors = input()
    if round(float(num_floors) == float(num_floors)) and num_floors !='1':
        num_floors = num_floors +'.0'
    elif num_floors != '1':
        var_dict['bathrooms_'+num_floors]
    print('Input Neighborhood:')
    neighborhood = input()
    if neighborhood != 'Auburn':
        var_dict[neighborhood] = 1
    print('Input Renovated (new, old or never):')
    ren = input()
    if ren == 'new':
        var_dict['new_ren'] = 1
    elif ren == 'old':
        var_dict['old_ren'] = 1
    print('Input Condition (Poor, Fair, Average, Good, Very Good):')
    condition = input()
    if condition in ['Poor', 'Fair', 'Good', 'Very Good']:
        var_dict['cond_'+condition] = 1  
    print('Input Grade Number:')    
    grade = input()
    if grade in ['5','6','7','8','9','10','11','12']:
        var_dict['cond_'+condition] = 1  
    print('Input Waterfront (y or n):')
    water = input()
    if water == 'y':
        var_dict['waterfront'] = 1
    print('Input Sqft Living Space:')
    sqft_living = float(input())
    var_dict['sqft_living'] = sqft_living
    print('Input Sqft Lot:')
    sqft_lot = float(input())
    var_dict['sqft_lot'] = sqft_lot
    print('Input Sqft Surronding Living Spaces:')
    sqft_living15 = float(input())
    var_dict['sqft_living15'] = sqft_living15
    print('Input Sqft Surrounding Lots:')
    sqft_lot15 = float(input())
    var_dict['sqft_lot15'] = sqft_lot15
    if scaler == 'log1p':
        for col in scaled_columns:
            if col != target:
                var_dict[col]=np.log1p(var_dict[col])
        array_values = np.array(list(var_dict.values()))
        prediction = np.sum(array_values * model.coef_) + model.intercept_
        if target in scaled_columns:
            prediction = np.expm1(prediction)
    elif scaler:
        z = list(zip(scaler.mean_, scaler.scale_))
        scale_dict = dict(zip(scaled_columns, z))
        for col in scaled_columns:
            if col != target:
                var_dict[col] = (var_dict[col]-scale_dict[col][0])/scale_dict[col][1]
        array_values = np.array(list(var_dict.values()))
        prediction = np.sum(array_values * model.coef_) + model.intercept_
        if target in scaled_columns:
            prediction = prediction*scale_dict[target][1] + scale_dict[target][0]    
    return round(prediction, ndigits=rnd)




