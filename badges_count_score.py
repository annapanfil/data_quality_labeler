from dataset_creator import MISSING_SYMBOLS
import pandas as pd
import numpy as np


def count_outliers_percentage_and_most_outliers_column(data: pd.DataFrame):
    ## Check outliers
    # For numerical values we use the same method as in box plot (outlier is more tham q3 + 1.5 IQR or less than q1 - 1.5 IQR)
    numeric_cols = data.select_dtypes(include=['number']).columns
    string_cols = data.select_dtypes(include=["string", "object"]).columns
    outliers_nums = []

    for col in numeric_cols:
        
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        
        iqr = q3-q1
        
        upper_bound = q3 + (1.5*iqr)
        lower_bound = q1 - (1.5*iqr)

        outliers_nums.append(np.sum((data[col] > upper_bound) | (data[col] < lower_bound)))

    # For string columns we look for rare values (less than 5% of the observations)    
    for col in string_cols:
        if not len(data[col].unique())/len(data[col]) > 0.5 and\
                (rare := data["category"].value_counts().min()/data.shape[0]) < 0.05: # rare category
            outliers_nums.append(rare)

    return sum(outliers_nums)/data.size, max(outliers_nums)/data.shape[0]


def count_dimminated_and_unique_columns(data: pd.DataFrame):
    # We also look for dominant values (more than 80% of the observations in column) and columns with unique values (eg. id, email), which may be not useful in further predictions.
    
    dominated_columns, unique_columns= 0, 0
    for col in data.columns:  
        if len(data[col].unique())/len(data[col]) > 0.5: # column with rather unique values
            unique_columns += 1      
        if data[col].value_counts().max()/data.shape[0] > 0.8: # dominant category
            dominated_columns += 1
            
    dominated_columns /= len(data.columns)
    unique_columns /= len(data.columns)

    return dominated_columns, unique_columns


def count_mishmashed(data: pd.DataFrame):
    ## Check mishmashed formats
    string_cols = data.select_dtypes(include=["string", "object"]).columns
    mishmashed_cases = []
    for col in string_cols:
        unique_in_data = len(data["category"].unique())
        truly_unique = len(data["category"].map(lambda x: x.lower() if not pd.isna(x) else x).unique())

        mishmashed_cases.append((unique_in_data - truly_unique)/truly_unique)
    return max(mishmashed_cases)


def count_score(dataset_scores: pd.DataFrame,
    weights = {
        "missing_percentage": 10, # many missing values is difficult to handle
        "most_missing_column": 2, # if 1 we had a column with huge amount of missing values, we'd have to drop it
        "duplication_percentage": 4, # many duplicates means less data
        "outliers_percentage": 2, # outliers may be removed or cause problems with predictions
        "most_outliers_column": 1,
        "unique_columns": 5, # if all columns are unique, we can't do much with it
        "dominated_columns": 3, # if a column has one dominant category, it may be not very useful
        "max_mishmashed_case": 1 # our data may be dirty and require a lot of cleaning
    })-> int:

    final_score = 0
    for name, score in dataset_scores.items():
        final_score += score * weights[name]

    final_score /= sum(weights.values())
    final_score = 1 - final_score # 1 is the best score, 0 – the worst
    return final_score


def data_preparation(data: pd.DataFrame):
    data.replace(MISSING_SYMBOLS, np.nan, inplace=True)
    data = data._convert(numeric=True, datetime=True).convert_dtypes()
    return data
