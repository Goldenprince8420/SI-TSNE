import matplotlib.pyplot as plt
import seaborn as sns


class DriverExplorer:
    def __init__(self, data):
        self.frame = data
        self.correlation = self.frame.corr()

    def show_heatmap(self):
        sns.heatmap(self.correlation)
        plt.show()

    def show_countplot(self, column):
        plt.figure(figsize = (10, 8))
        sns.countplot(x = self.frame[column])
        plt.show()

    def show_histogram(self, column):
        plt.figure(figsize=(10, 8))
        sns.histplot(self.frame[column])
        plt.show()


class HistPlotter:
    def __init__(self, data):
        self.frame = data
        self.columns = None

    def plot_for_all_columns(self):
        plt.figure(figsize = (10, 8))
        pal_idx = 1
        for column in self.frame.columns:
            sns.histplot(self.frame[column], palette = sns.color_palette("hls", pal_idx))
            plt.show()
            pal_idx += 1
        print("Done!!")

    def plot_for_columns(self, columns = None):
        plt.figure(figsize = (10, 8))
        pal_idx = 1
        for column in columns:
            sns.histplot(self.frame[column], palette = sns.color_palette("hls", pal_idx))
            plt.show()
            pal_idx += 1
        print("Done!!")
