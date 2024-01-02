import numpy as np
import pandas as pd
import streamlit as st 
from collections import defaultdict
from badges_count_score import *

def count_scores(data: pd.DataFrame):

    dataset_scores = defaultdict(lambda: 0)
    dataset_scores["missing_percentage"] = data.isna().sum().sum()/data.size
    dataset_scores["most_missing_column"] = data.isna().sum().max()/data.shape[0]
    dataset_scores["duplication_percentage"] = sum(data.duplicated())/ data.shape[0]
    dataset_scores["outliers_percentage"], dataset_scores["most_outliers_column"] = count_outliers_percentage_and_most_outliers_column(data)
    dataset_scores["dominated_columns"], dataset_scores["unique_columns"] = count_dimminated_and_unique_columns(data)
    dataset_scores["max_mishmashed_case"] = count_mishmashed(data)

    return dataset_scores, count_score(dataset_scores)


def show_badges(dataset_scores: dict):
    for badge_name, budge_score in dataset_scores.items():
        st.markdown(f"![DQ badge](https://img.shields.io/badge/{badge_name}-{round(budge_score, 2):.2}-{'red' if budge_score < 0.5 else 'blue'})")

    st.markdown('## Total dataset quality score')
    centered_image_html = f'<div style="display: flex; justify-content: center;"><img src="https://img.shields.io/badge/{"dataset_quality_score"}-{round(dataset_final_quality_score, 2):.2}-{"red" if dataset_final_quality_score < 0.5 else "blue"}" width="200" height="40"></div>'
    st.markdown(centered_image_html, unsafe_allow_html=True)


if __name__ == "__main__":

    st.title("Rate your dataset")
    st.write("Upload a dataset in csv format")
    uploaded_file = st.file_uploader("Choose a file", type="csv") # TODO: add on_change

    if uploaded_file is not None:
        data = data_preparation(pd.read_csv(uploaded_file))

        dataset_scores, dataset_final_quality_score = count_scores(data)

        show_badges(dataset_scores)


