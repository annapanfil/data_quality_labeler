import numpy as np
import pandas as pd
import streamlit as st 
from collections import defaultdict
from dataset_creator import MISSING_SYMBOLS


def count_scores(data: pd.DataFrame):
    dataset_scores = defaultdict(lambda: 0)

    ## Check if there are any missing values
    data.replace(MISSING_SYMBOLS, np.nan, inplace=True)
    data.isna().sum()
    data = data._convert(numeric=True, datetime=True).convert_dtypes()

    dataset_scores["missing_percentage"] = data.isna().sum().sum()/data.size
    dataset_scores["most_missing_column"] = data.isna().sum().max()/data.shape[0]

    data.results1.astype("float")

    ## Check duplicates

    dataset_scores["duplication_percentage"] = sum(data.duplicated())/ data.shape[0]

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

    dataset_scores["outliers_percentage"] = sum(outliers_nums)/data.size
    dataset_scores["most_outliers_column"] = max(outliers_nums)/data.shape[0]


    # We also look for dominant values (more than 80% of the observations in column) and columns with unique values (eg. id, email), which may be not useful in further predictions.
    for col in data.columns:  
        if len(data[col].unique())/len(data[col]) > 0.5: # column with rather unique values
            dataset_scores["unique_columns"] += 1      
        if data[col].value_counts().max()/data.shape[0] > 0.8: # dominant category
            dataset_scores["dominated_columns"] += 1
            
    dataset_scores["dominated_columns"] /= len(data.columns)
    dataset_scores["unique_columns"] /= len(data.columns)


    ## Check mishmashed formats
    mishmashed_cases = []
    for col in string_cols:
        unique_in_data = len(data["category"].unique())
        truly_unique = len(data["category"].map(lambda x: x.lower() if not pd.isna(x) else x).unique())

        mishmashed_cases.append((unique_in_data - truly_unique)/truly_unique)

    dataset_scores["max_mishmashed_case"] = max(mishmashed_cases)

    ## Aggregate scores
    weights = {
        "missing_percentage": 10, # many missing values is difficult to handle
        "most_missing_column": 2, # if 1 we had a column with huge amount of missing values, we'd have to drop it
        "duplication_percentage": 4, # many duplicates means less data
        "outliers_percentage": 2, # outliers may be removed or cause problems with predictions
        "most_outliers_column": 1,
        "unique_columns": 5, # if all columns are unique, we can't do much with it
        "dominated_columns": 3, # if a column has one dominant category, it may be not very useful
        "max_mishmashed_case": 1 # our data may be dirty and require a lot of cleaning
    }

    assert weights.keys() == dataset_scores.keys()

    final_score = 0
    for name, score in dataset_scores.items():
        final_score += score * weights[name]

    final_score /= sum(weights.values())
    final_score = 1 - final_score # 1 is the best score, 0 – the worst

    dataset_scores["dataset_quality_score"] = final_score

    return dataset_scores

if __name__ == "__main__":
    # read data
    st.title("Rate your dataset")
    st.write("Upload a dataset in csv format")
    uploaded_file = st.file_uploader("Choose a file", type="csv") # TODO: add on_change

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        dataset_scores = count_scores(data)

        # Create badges
        print(dataset_scores)

        # print("To add badges paste this to your readme.md file:")
        for badge_name, budge_score in dataset_scores.items():
            st.markdown(f"![DQ badge](https://img.shields.io/badge/{badge_name}-{budge_score:.2}-blue)")



