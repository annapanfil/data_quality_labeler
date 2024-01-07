import numpy as np
import pandas as pd
import streamlit as st 
from collections import defaultdict
from badges_count_score import *

weights = {
        "missing_percentage": 10, # many missing values is difficult to handle
        "most_missing_column": 2, # if 1 we had a column with huge amount of missing values, we'd have to drop it
        "duplication_percentage": 4, # many duplicates means less data
        "outliers_percentage": 2, # outliers may be removed or cause problems with predictions
        "most_outliers_column": 1,
        "unique_columns": 5, # if all columns are unique, we can't do much with it
        "dominated_columns": 3, # if a column has one dominant category, it may be not very useful
        "max_mishmashed_case": 1, # our data may be dirty and require a lot of cleaning
        'correlation_numerical' : 4,
        'correlation_categorical' : 1,
        'missing documentation' : 2
    }

def count_scores(data: pd.DataFrame, document: str):
    dataset_scores = defaultdict(lambda: 0)
    dataset_scores["missing_percentage"] = data.isna().sum().sum()/data.size
    dataset_scores["most_missing_column"] = data.isna().sum().max()/data.shape[0]
    dataset_scores["duplication_percentage"] = sum(data.duplicated())/ data.shape[0]
    dataset_scores["outliers_percentage"], dataset_scores["most_outliers_column"] = count_outliers_percentage_and_most_outliers_column(data)
    dataset_scores["dominated_columns"], dataset_scores["unique_columns"] = count_dimminated_and_unique_columns(data)
    dataset_scores["max_mishmashed_case"] = count_mishmashed(data)
    dataset_scores["correlation_numerical"] = count_correlation_badges(data)
    dataset_scores["correlation_categorical"] = count_correlation_badges_categorical(data)
    dataset_scores["missing documentation"] = count_documentation_detail(document)

    return dataset_scores, count_score(dataset_scores, weights=weights)


def display_sliders():
    for badge_name, budge_weight in weights.items():
        weight = st.slider(f"Set {badge_name} badge weight",0, 10, budge_weight, 1)
        weights[badge_name] = weight

    return count_score(dataset_scores, weights=weights)


def show_badges(dataset_scores: dict):
    st.markdown("## Certin badge scores")
    for badge_name, budge_score in dataset_scores.items():
        st.markdown(f"![DQ badge](https://img.shields.io/badge/{badge_name}-{round(budge_score, 2)}-{'red' if budge_score > 0.5 else 'blue'})")

    st.markdown('## Total dataset quality score')
    dataset_final_quality_score = display_sliders()
    centered_image_html = f'<div style="display: flex; justify-content: center;"><img src="https://img.shields.io/badge/{"dataset_quality_score"}-{round(dataset_final_quality_score, 2):.2}-{"red" if dataset_final_quality_score < 0.5 else "blue"}" width="250" height="40"></div>'
    st.markdown(centered_image_html, unsafe_allow_html=True)


if __name__ == "__main__":
    st.markdown("<h1 style='text-align: center;'>Rate your dataset ⭐️</h1>", unsafe_allow_html=True)
    st.markdown("#### Upload a dataset in csv format")
    uploaded_file = st.file_uploader("Choose a file", type="csv") # TODO: add on_change
    st.markdown("#### Upload an documentation in txt format")
    document = st.file_uploader("Choose a documentation file", type="txt")

    if uploaded_file is not None:
        data = data_preparation(pd.read_csv(uploaded_file))
        dataset_scores, _ = count_scores(data, document)
        show_badges(dataset_scores)
