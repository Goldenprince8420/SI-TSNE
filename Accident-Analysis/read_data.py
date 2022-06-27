import pandas as pd


class ReadData:
    def __init__(self, path):
        self.path = path
        self.frame = None

    def read(self):
        self.frame = pd.read_csv(self.path)

    def get_data(self):
        self.read()
        return self.frame


class DataStat:
    def __init__(self, data):
        self.frame = data
        self.num_rows = len(self.frame)
        self.num_cols = len(self.frame.columns)
        self.correlation = self.frame.corr()
        self.description = self.frame.describe()

    def get_information(self):
        self.frame.info()
        return

    def get_correlation(self):
        return self.correlation

    def get_description(self):
        return self.description

    def print_dimensions(self):
        print("Number of Rows: {}\nNumber of Columns: {}".format(self.num_rows, self.num_cols))

