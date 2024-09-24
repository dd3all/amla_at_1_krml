import pandas as pd
import os
from joblib import load, dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce

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

    y = df.pop(target_col)
    x = df

    return x, y

def convert_to_datetime(df, col):  
    """
    Convert a column to pandas datetime datatype

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    col : str
        Name of the column to be converted

    Returns
    -------
    pd.DataFrame
        Dataframe with the target column converted to datetime
    """

    df[col] = pd.to_datetime(df[col])

    return df

def year_month_weekend(df, date_col, year='year', month='month', is_weekend='is_weekend'):
    """
    Extract new 'year', 'month', 'is_weekend' columns for a df from a given datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    date_col : str
        Name of the datetime column
    year : str
        Name of the year column to be created. Default 'year'
    month : str
        Name of the month column to be created. Default 'month'
    is_weekend : str
        Name of the is_weekend column to be created. Default 'is_weekend'

    Returns
    -------
    pd.DataFrame
        Dataframe with the newly created columns
    """
    
    # Extract year
    df[year] = df[date_col].dt.year

    # Extract month
    df[month] = df[date_col].dt.month

    # Extract whether the date is a weekday (Saturday=5, Sunday=6)
    df[is_weekend] = df[date_col].dt.weekday >= 5

    return df

def join_dfs(df_1, df_2, col, how):
    """
    Join two dataframes based on desired conditions

    Parameters
    ----------
    df_1 : pd.DataFrame
        Dataframe 1 
    df_2 : pd.DataFrame
        Dataframe 2
    col : str or list
        Name(s) of the column to join the dfs on
    how : str
        Type of the join

    Returns
    -------
    pd.DataFrame
        New joined dataframe
    """

    new_df = df_1.merge(df_2, on=col, how=how)

    return new_df

def melt_df(df, cols_to_keep, col_name_melted, col_name_value):
    """
    Join two dataframes based on desired conditions

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to melt 
    cols_to_keep : list
        List of columns to keep
    col_name_melted : str
        New column name of the melted columns
    col_name_value : str
        New column name of the melted values.

    Returns
    -------
    pd.DataFrame
        New joined dataframe
    """

    new_df = df.melt(id_vars=cols_to_keep, var_name=col_name_melted, value_name=col_name_value)

    return new_df

def create_region(column):
    """
    Creates a new column for states.

    Parameters
    ----------
    col : str
        Name of the column to extract the new columns from

    Returns
    -------
    str
        Encoded state id's based on stores.
    """

    if column.startswith('CA'):
        return 'CA'
    elif column.startswith('TX'):
        return 'TX'
    elif column.startswith('WI'):
        return 'WI'
    else:
        return None
    
def assign_region(df, base_col, new_col='state'):
    """
    Assign create region function based on a reference column and creates the new state colum.

    Parameters
    ----------
    df : pd.DataFrame
        Df in question
    base_col : str
        Existing column on which the new column shall be extracted.
    new_col : str
        Name of the new column. Default: 'state'

    Returns
    -------
    pd.DataFrame
        New dataframe with created state column
    """

    df[new_col] = df[base_col].apply(create_region)

    return df

def create_revenue(df, new_col='revenue', col1='sold_amount', col2='sell_price'):
    """
    Create new revenue column by multiplying two columns amount and sell_price

    Parameters
    ----------
    df : pd.DataFrame
        Df in question
    new_col : str
        New column name to be created. Default: revenue
    col1 : str
        Name of the first column for multiplication. Default: 'sold_amount'
    col2 : str
        Name of the second column for multiplication. Default: 'sell_price'

    Returns
    -------
    pd.DataFrame
        New dataframe with created state column
    """

    df[new_col] = df[col1] * df[col2]

    return df

def BaseNencoder(X_train, X_val, cols, base=10):
    """
    Apply BaseN Encoder to train and validation sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        X_train
    X_val : pd.DataFrame
        X_val
    cols : list
        List of the columns to be encoded 
    Base : int
        Integer value of number of bases. Default: 10

    Returns
    -------
    X_train : pd.DataFrame
        Encoded X_train
    X_val : pd.DataFrame
        Encoded X_val
    encoder : model
        Fit and trained BaseN transformer
    """

    encoder = ce.BaseNEncoder(cols=cols, base=base)

    # Fit and transform X_train
    X_train = encoder.fit_transform(X_train)

    # Transform X_val
    X_val = encoder.transform(X_val)

    return X_train, X_val, encoder

def label_encoder(X_train, X_val, cols):
    """
    Apply Label Encoder to train and validation sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        X_train
    X_val : pd.DataFrame
        X_val
    cols : list
        List of the columns to be encoded.

    Returns
    -------
    X_train : pd.DataFrame
        Encoded X_train
    X_val : pd.DataFrame
        Encoded X_val
    encoder : model
        Fit and trained BaseN transformer
    """
    encoder = ce.OrdinalEncoder(cols=cols)

    # Apply label encoder to train
    X_train = encoder.fit_transform(X_train)

    # Transform X_val
    X_val = encoder.transform(X_val)

    return X_train, X_val, encoder



def apply_MinMax(X_train, X_val):
    """
    Apply MinMax Scaler to train and validation sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        X_train
    X_val : pd.DataFrame
        X_val

    Returns
    -------
    X_train : pd.DataFrame
        Encoded X_train
    X_val : pd.DataFrame
        Encoded X_val
    scaler : model
        Fit and trained BaseN transformer
    """
    # Instantiate a scaler
    scaler = MinMaxScaler()

    # Fit transform X_train
    X_train = scaler.fit_transform(X_train)

    # Transform X_val
    X_val = scaler.transform(X_val)

    return X_train, X_val, scaler

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



