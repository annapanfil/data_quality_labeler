from dataset_creator import MISSING_SYMBOLS
import pandas as pd
import numpy as np
from search_strings import *
from scipy.stats import chi2_contingency


def count_documentation_detail(doc_dest: str):
    if doc_dest is None:
        return 1
    
    content = doc_dest.read().decode("utf-8").lower() # Lowercase to avoid inconsistensies
    context = 0
    for search_string in search_strings:
        for i in search_string:
            if i in content:
                context = context + 1
                break
    
    return 1 - (context / len(search_strings))


def count_correlation_badges_categorical(data: pd.DataFrame):
    categorical_data = data.select_dtypes(include=['object', 'category', 'string'])    
    dependent, independent = 0, 0
    for i, col in enumerate(categorical_data.columns):
        for j, col2 in enumerate(categorical_data.columns):
            if i != j:
                CrosstabResult=pd.crosstab(index=categorical_data[col],columns=categorical_data[col2])
                chi2, p, dof, expected = chi2_contingency(CrosstabResult)
                alpha = 0.01
                if p <= alpha:
                    dependent += 1
                else:
                    independent += 1
    
    return 1 - independent/(independent+dependent)  # 1 the worst, 0 the best


def count_correlation_badges(data: pd.DataFrame):
    # Check correlation between columns
    # We check correlation between columns and if it's higher than 0.8, we mark it as a bad thing
    numeric_data = data.select_dtypes(include=['number'])
    corr = numeric_data.corr()
    np.fill_diagonal(corr.values, 0)
    count_of_highly_correlated_columns = (abs(corr) > 0.8).sum().sum()
    return count_of_highly_correlated_columns / 2*len(numeric_data.columns) # 1 the worst, 0 theh best 


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
                (rare := data[col].value_counts().min()/data.shape[0]) < 0.05: # rare category
            outliers_nums.append(rare)

    return sum(outliers_nums)/data.size, max(outliers_nums)/data.shape[0]


def count_dimminated_and_unique_columns(data: pd.DataFrame):
    # We also look for dominant values (more than 80% of the observations in column) and columns with unique values (eg. id, email), which may be not useful in further predictions.
    
    dominated_columns, unique_columns= 0, 0
    for col in data.columns:  
        if len(data[col].unique())/len(data[col]) > 0.5: # column with rather unique values
            unique_columns += 1     
        # smaller threshold 
        if data[col].value_counts().max()/data.shape[0] >= 0.8: # dominant category
            dominated_columns += 1
            
    dominated_columns /= len(data.columns)
    unique_columns /= len(data.columns)

    return dominated_columns, unique_columns


def count_mishmashed(data: pd.DataFrame):
    ## Check mishmashed formats
    string_cols = data.select_dtypes(include=["string", "object"]).columns
    mishmashed_cases = []
    for col in string_cols:
        unique_in_data = len(data[col].unique())
        truly_unique = len(data[col].map(lambda x: x.lower() if not pd.isna(x) else x).unique())

        mishmashed_cases.append((unique_in_data - truly_unique)/truly_unique)
    return max(mishmashed_cases)


def count_score(dataset_scores: pd.DataFrame, weights)-> int:

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
