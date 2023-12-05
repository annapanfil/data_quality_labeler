from faker import Faker
import pandas as pd
import random

MISSING_SYMBOLS = [None, "null", "", "NULL", "N/A", "n/a", "NA", "na", "NaN", "nan", "None", "none", "NONE"]


def create_dataset(dataset_size = 100, 
                   columns=["name", "surname", "birthdate", "results1", "results2", "category", "email"],
                   missing_percentage = 0.1,
                   output_file="dataset.csv"          
):
    """
    Create a dataset with random values and missing values
    """

    if type(missing_percentage) == float:
        missing_percentage = [missing_percentage] * len(columns)

    fake = Faker()

    data = pd.DataFrame(columns=columns)

    data["name"],data["surname"] = zip(*[fake.name().split() for _ in range(dataset_size)])
    data["birthdate"] = [fake.date_of_birth() for _ in range(dataset_size)]
    data["results1"] = [random.randint(0,100) for _ in range(dataset_size)]
    data["results2"] = [random.random() for _ in range(dataset_size)]
    data["category"] = random.choices(["A", "B", "C"], k=dataset_size)
    data["email"] = [fake.email() for _ in range(dataset_size)]

    ## mishmash a bit
    # missing
    for column_no, missing_p in enumerate(missing_percentage):
        data.iloc[data.sample(frac=missing_p).index, column_no] = random.choices(MISSING_SYMBOLS, k=int(missing_p*dataset_size))

    # future dates in birthdate

    # outliers

    # lowercase categories

    # not int results

    # ...

    data.to_csv(output_file, index=False)

    return output_file