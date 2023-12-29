from faker import Faker
import pandas as pd
import numpy as np
import random

MISSING_SYMBOLS = [None, "null", "", "NULL", "N/A", "n/a", "NA", "na", "NaN", "nan", "None", "none", "NONE"]

class FakeDataset:
    def __init__(self, dataset_size = 100, 
                   columns=["name", "surname", "birthdate", "results1", "results2", "category", "email"]):            
        """
        Create a dataset with random values and missing values
        """

        fake = Faker()

        self.data = pd.DataFrame(columns=columns)

        self.data["name"], self.data["surname"] = zip(*[fake.name().split() for _ in range(dataset_size)])
        self.data["birthdate"] = [fake.date_of_birth() for _ in range(dataset_size)]
        self.data["results1"] = [random.randint(0,100) for _ in range(dataset_size)]
        self.data["results2"] = [random.normalvariate(0, 1) for _ in range(dataset_size)]
        self.data["category"] = random.choices(["A", "B", "C"], k=dataset_size)
        self.data["email"] = [fake.email() for _ in range(dataset_size)]

    def add_missing(self, missing_percentage):

        # because each column can have different one
        if type(missing_percentage) == float:
            missing_percentage = [missing_percentage] * self.data.shape[1]

        for column_no, missing_p in enumerate(missing_percentage):
            missing = random.choices(MISSING_SYMBOLS, k=np.ceil(missing_p * self.data.shape[0]).astype("int"))
            self.data.iloc[self.data.sample(frac=missing_p).index, column_no] = missing
        return self
    
    def add_duplicates(self, duplicate_percentage):
        self.data = self.data.append(self.data.iloc[self.data.sample(frac=duplicate_percentage).index, :])

        return self
    
    def add_outliers_above(self, outlier_percentage, column="results1"):
        column_no = self.data.columns.get_loc(column)
        max_value = self.data[column].max()

        outliers = random.choices(range(int(max_value * 5), int(max_value * 6)), k=np.ceil(outlier_percentage * self.data.shape[0]).astype("int"))
        self.data.iloc[self.data.sample(frac=outlier_percentage).index, column_no] = outliers
                
        return self 

    def to_csv(self, filename):
        self.data.to_csv(filename, index=False)
        # future dates in birthdate

        # outliers

        # lowercase categories

        # not int results

        # ...


if __name__ == "__main__":
    filename = 'dataset.csv'

    dataset = FakeDataset(dataset_size = 100)\
            .add_outliers_above(outlier_percentage = 0.1)\
            .add_duplicates(duplicate_percentage = 0.15)\
            .add_missing(missing_percentage = 0.1)\
            # .to_csv(filename)

    print(dataset.data.head())