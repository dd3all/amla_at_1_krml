import pandas as pd
import os
from joblib import load, dump
from sklearn.preprocessing import StandardScaler

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target



def load_sets(path='../data/processed/'):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    
    Raises
    ------
    FileNotFoundError
        If a specified file is not found in the directory
    """
    import os.path

    x_train = pd.read_csv(os.path.join(path, 'x_train.csv'))
    x_val   = pd.read_csv(os.path.join(path, 'x_val.csv'))

    y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))
    y_val   = pd.read_csv(os.path.join(path, 'y_val.csv'))

    return x_train, y_train, x_val, y_val


def apply_poly(data, path='../models/'):
    """
    Load saved Polynominal transformer and apply it to the provided dataset

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to be transformed using Poly transformer
    path : str
        Path to the saved poly transformer. (Default: ../models/)
    
        
    Returns
    -------
    np.ndarray
        Transformed data.
    """

    poly = load(os.path.join(path, 'poly.joblib'))

    tf_data = poly.transform(data)

    return tf_data



def train_standard_scaler(x_train, path='../models/'):
    """
    Train a standard scaler and save it to the specified directory

    Parameters
    ----------
    x_train : pd.DataFrame
        The training feature set that needs to be fitted to the scaler.
    path : str
        Path to save the trained standard scaler. (Default: ../models/)

    Returns
    -------
    scaler : scikit-learn Scaler object
        Fitted and trained Standard Scaler
    """

    scaler = StandardScaler()

    scaler.fit(x_train)

    if not os.path.exists(path):
        os.makedirs(path)

    dump(scaler, os.path.join(path, 'standard_scaler.joblib'))

    return scaler


def apply_standard_scaler(features, df, num_cols, path='../models/'):
    """
    Load saved Standard Scaler and apply it to the provided dataset

    Parameters
    ----------
    features : pd.DatFrame
        Data frame of categorical features that have already been encoded
    df : pd.DataFrame
        The full dataframe to be transformed using Standard Scaler
    num_cals : list
        The list of the numerical columns of the dataframe
    path : str
        Path to the saved Standard Scaler. (Default: ../models/)
    
        
    Returns
    -------
    np.ndarray
        Transformed data.
    """

    scaler = load(os.path.join(path, 'scaler.joblib'))

    features[num_cols] = scaler.transform(df[num_cols])

    return features


def num_cat_cols(df):
    """
    Separate lists of numerical columns and categorical columns from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to separate the numeric columns and categorical columns of 
    
        
    Returns
    -------
    num_cols : list
        List of numerical columns
    cat_cols : list
        List of categorical columns
    """
    num_cols = list(df.select_dtypes('number').columns)
    cat_cols = list(set(df.columns) - set(num_cols))

    return num_cols, cat_cols


def apply_ohe(df, cat_cols,path='../models/'):
    """
    Apply trained OneHotEncoder to the categorical columns of a data frame

    Parameters
    ----------
    df : pd.DataFrame
        The data frame of which the categorical columns to be encoded.
    
    cat_cols : list
        The list of the categorical columns to be encoded.
    path : str
        The directory path where the encoder is stored.(Default: ../models/)

    Returns
    -------
    features : pd.DataFrame
        A data frame with categorical columns one-hot-encoded.
    """
    
    ohe = load(os.path.join(path, 'ohe.joblib'))

    features = ohe.transform(df[cat_cols])

    features = pd.DataFrame(features, columns=ohe.get_feature_names_out())

    return features


def drop_cols(df, cols):
    """
    Drop columns from a data frame

    Parameters:
    -----------
    df : pd.DataFrame
        The data frame columns of which are to be dropped
    cols : List
        List of the column names to be dropped.

        
    Returns:
    --------
    output : pd.DataFrame
        The result dataframe with the specified columns dropped.
    """
    
    output = df.drop(cols, axis=1)

    return output


def null_impute(df, col_names, sub=None):
    """
    Imputes null values of a column based on user selection. (Mean or any given string value)


    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe of which the column should be imputed.
    col_names : list
        list of column names to be imputed
    sub : 'str'
        Value to substitute the null values. (Default: None)


    Returns:
    --------
    df : pd.DataFrame
        Dataframe with the specified columns imputed
    """

    for col_name in col_names:
        if sub and sub != 'mean':
            df[col_name].fillna(sub, inplace=True)
        elif sub is None or sub == 'mean':
            mean = df[col_name].mean()
            df[col_name].fillna(mean, inplace=True)

    return df


def generate_results_csv(df, probas, name='predictions', path='./'):
    """
    Generates a result csv file as requested and prompts a success notification.

    Parameters
    ----------
    df : pd.DataFrame
        Default dataframe which will be used to capture player ID's.
    probas : list or np.array
        List of predicted probabilities of each player being drafted.
    name : str
        Name of the results csv file. (Default: predictions)
    path : str
        Directory path where the result csv file is to be stored.

    Returns
    -------
    None

    """

    player_id = df['player_id']

    results = pd.DataFrame({
    'player_id': player_id,
    'drafted': probas
    })

    results['drafted'] = results['drafted'].clip(upper=0.99).round(2)

    full_path = os.path.join(path, f'{name}.csv')

    results.to_csv(full_path, index=False)

    print(f'Results file has been successfully extracted to {full_path}.')



