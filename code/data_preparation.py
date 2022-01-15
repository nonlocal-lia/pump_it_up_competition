import numpy as np
import pandas as pd
import datetime
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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


def impute_by_area(df, areas_lg_to_sm, col_name):
    """
    Imputes the nan values of the column named using the means of the listed areas.
    The mean value of the smallest area is favored.

    Arg:
        df(pdDataFrame): A dataframe with a column to imput values of
        areas_lg_to_sm(array): a list of the column names(str) of geographic areas 
                                to use ordered from largest to smallest in area
        col_name(str): the name of the column you desire to impute values of

    Return:
        df(pdDataFrame): The dataframce with extra columns for the imputed data,
                        includes columns for each area size as well
    """
    areas = areas_lg_to_sm.copy()
    i = 0
    for area in areas_lg_to_sm:
        means_pop = df.groupby(areas)[col_name].mean().reset_index()
        means_pop = means_pop.rename(
            columns={col_name: col_name+'_'+areas[-1]})
        df = df.merge(means_pop, how='left', on=areas)
        if i == 0:
            df['imputed_'+col_name] = np.where(df[col_name].isna(),
                                               df[col_name+'_'+areas[-1]],
                                               df[col_name])
        else:
            df['imputed_'+col_name] = np.where(df['imputed_'+col_name].isna(),
                                               df[col_name+'_'+areas[-1]],
                                               df['imputed_'+col_name])
        areas = areas[:-1]
    # Imputes the total mean for any remaining uncaptured values
    df['imputed_'+col_name].fillna(df['imputed_'+col_name].mean())
    return df


def clean_test_data(df):
    """
    Cleans data from the pump it up competition for modeling or prediction.

    Arg:
        df(pdDataFrame): A dataframe with data from the pump it up competition

    Return:
        clean(pdDataFrame): The dataframe cleaned for use in modeling or prediction.
    """
    # Drop unused columns
    dropped_cols = ['id', 'wpt_name', 'num_private',
                    'region', 'recorded_by', 'scheme_name',
                    'extraction_type_group', 'payment_type',
                    'group_value', 'quantity_group',
                    'source_type', 'waterpoint_type']
    clean = df.drop(columns=dropped_cols)

    # Fix input errors
    replace_dict = {'commu':  'community',
                    'gover': 'government',
                    'central government': 'government',
                    'government of tanzania': 'government',
                    'gove': 'government',
                    'worldvision': 'world vision',
                    'private individual': 'private',
                    'privat': 'private',
                    'priv': 'private'}
    error_cols = ['funder', 'installer', 'subvillage']
    for col in error_cols:
        clean[col] = clean[col].str.lower()
    clean[error_cols] = clean[error_cols].replace(replace_dict)

    # Impute Mode for Permit, Meeting
    clean['missing_meeting'] = missing_indicator(clean, 'public_meeting')
    clean['public_meeting'] = clean['public_meeting'].fillna(
        clean['public_meeting'].mode()[0])
    clean['missing_permit'] = missing_indicator(clean, 'permit')
    clean['permit'] = clean['permit'].fillna(clean['permit'].mode()[0])

    # Impute Scheme Management from Management
    idx = clean[clean['scheme_management'].isna()].index
    sm_column_num = clean.columns.get_loc('scheme_management')
    man_column_num = clean.columns.get_loc('management')
    clean.iloc[idx, sm_column_num] = clean.iloc[idx, man_column_num]

    # Fill Remaining Nan with 'not_known'
    clean = clean.fillna('not known')

    # Replace Zeros in Columns where it is clearly missing data to allow imputing
    zero_cols = ['construction_year', 'longitude', 'population', 'gps_height']
    clean[zero_cols] = clean[zero_cols].replace(0, np.nan)
    # Latitude has a zero of -2.000000e-08
    clean['latitude'] = clean['latitude'].replace(-2.000000e-08, np.nan)

    # Impute construction_year
    encoded_df = clean.copy()
    categorical_cols = clean.select_dtypes(
        include=['object', 'datetime64']).columns
    encoded_df[categorical_cols] = encoded_df[categorical_cols].apply(
        LabelEncoder().fit_transform)
    imp = IterativeImputer(random_state=42)
    imp.fit(encoded_df)
    encoded_df = imp.transform(encoded_df)
    clean.loc[:, 'construction_year'] = encoded_df[:,
                                                   clean.columns.get_loc('construction_year')]

    # Engineer pump_age column
    clean['recorded_year'] = pd.DatetimeIndex(clean['date_recorded']).year
    clean['pump_age'] = clean['recorded_year'] - clean['construction_year']

    # Engineer season column
    clean['month'] = pd.DatetimeIndex(clean['date_recorded']).month
    season_dict = {1: 'short dry', 2: 'short dry', 3: 'long rain',
                   4: 'long rain', 5: 'long rain', 6: 'long dry',
                   7: 'long dry', 8: 'long dry', 9: 'long dry',
                   10: 'long dry', 11: 'short rain', 12: 'short rain'}
    clean['season'] = clean['month'].replace(season_dict)

    # Drop Added Columns as well as data_recorded and construction year
    clean = clean.drop(
        columns=['month', 'recorded_year', 'construction_year', 'date_recorded'])

    # Impute Population, Height and Lat-Long by location
    areas_lg_to_sm = ['region_code', 'district_code', 'ward', 'subvillage']
    impute_by_area_cols = ['population', 'longitude', 'latitude', 'gps_height']
    for col in impute_by_area_cols:
        clean = impute_by_area(clean, areas_lg_to_sm, col)

    # Drop Extra Columns added by function
    labels = impute_by_area_cols.copy()
    for label in areas_lg_to_sm:
        for col in impute_by_area_cols:
            labels.append(col + '_' + label)
    clean = clean.drop(columns=labels)

    return clean
