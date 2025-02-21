import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats


def load_data(filepath):
    """simply returning pd.read_csv"""
    return pd.read_csv(filepath)


def handle_missing_values(df):
    """filling NAs with the mean of each column. will break if the df has missing values in string columns"""
    return df.fillna(df.mean())


def remove_outliers(df, z_scores: float):
    """Removes rows which are beyond z-scores threshold"""
    # the tutorial says we can go with df_no_outliers = df_filled[(z_scores < 3).all(axis=1)]
    # however, this didn't work so I wrote the loop below (ugly, but works)
    for column, values in df.items():
        try:
            column_name = f"{str(column)}-z_scores"
            # adding a z-scores column for each numeric variable
            df[column_name] = np.abs(stats.zscore(df[column]))
            # filtering the db to remove rows that are >= 3 std deviations
            df = df.loc[df[column_name] < z_scores]
            df = df.drop(column_name, axis=1)
        except:
            print(f"can't process column \"{column}\", skipping")
    return df


def cap_column_quantile(df, column_name, quantile):
    """receives a df, a column name in that df, and a quantile.
    returns the df with the provided column capped to the provided quantile
    """
    upper_limit = df[column_name].quantile(quantile)
    df[column_name] = np.where(
        df[column_name] > upper_limit, upper_limit, df[column_name])
    return df


def scale_data(df):
    """scaling data so all features contribute equally to the model"""
    df_temp = df.select_dtypes(
        exclude="object")  # here we can also use include=[np.number]
    for column in df_temp.columns:
        scaled_column = f"scaled-{column}"
        scaler = StandardScaler()
        df[scaled_column] = scaler.fit_transform(
            pd.DataFrame(df_temp[column]))
    # we can also go with this way more elegant solution:
    # scaler = StandardScaler()
    # df_no_outliers[df_no_outliers.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df_no_outliers.select_dtypes(include=[np.number]))
    return df


def encode_categorical(df, categorical_columns):
    """Convert categorical variables into a numerical format that machine learning algorithms can process"""
    return pd.get_dummies(df, columns=categorical_columns)


def save_data(df, output_filepath):
    """saves the df in csv format"""
    df.to_csv(output_filepath, index=False)


if __name__ == "main":
    # Example usage:
    df = load_data('your_dataset.csv')
    df = handle_missing_values(df)
    df = remove_outliers(df, 3)
    df = scale_data(df)
    df = encode_categorical(df, ['categorical_column_name'])
    save_data(df, 'cleaned_preprocessed_data.csv')
