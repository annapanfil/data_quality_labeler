# Data Quality labeler

An app to automatically check dataset quality and create a badge for it. Created as a final project for biomedical data science on UPV

Includes also fake dataset creator.

## Example badges
![DQ Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fannapanfil%2Fdata_quality_labeler%2Fmain%2F./badge_data.json&query=%24.missing_percentage&label=missing_percentage)

![DQ Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fannapanfil%2Fdata_quality_labeler%2Fmain%2F./badge_data.json&query=%24.most_missing_column&label=most_missing_column)


Badges created thanks to [shields.io](https://shields.io/badges/dynamic-json-badge)

## Run app
To run this app, copy and paste below commends. 
```bash
$ git clone https://github.com/annapanfil/data_quality_labeler/tree/main
$ git checkout web_app
$ conda create --name your_env_name
$ conda activate your_env_name
$ pip install pandas=1.5.3
$ pip install streamlit
$ pip install faker
$ pip install numpy
$ streamlit run main.py  
```

## Screen shots from app 
![Alt text](image.png)
![Alt text](image-1.png)
![Alt text](image-2.png)
![Alt text](image-3.png)