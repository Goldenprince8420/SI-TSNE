import numpy as np
import pandas as pd
from scipy import stats


class AdvancedAnalyserDriver:
    def __init__(self, data):
        self.frame = data
        self.columns = list(self.frame.columns)

    def pearson_analysis(self):
        r_values = []
        p_values = []
        for column1 in self.frame.columns:
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                r, p = stats.pearsonr(x, y)
                temp_r.append(r)
                temp_p.append(p)
            r_values.append(temp_r)
            p_values.append(temp_p)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        return df_r, df_p
    
    def spearman_analysis(self):
        r_values = []
        p_values = []
        for column1 in self.frame.columns:
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                r, p = stats.spearmanr(x, y)
                temp_r.append(r)
                temp_p.append(p)
            r_values.append(temp_r)
            p_values.append(temp_p)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        return df_r, df_p

    def point_biserial_analysis(self):
        r_values = []
        p_values = []
        for column1 in self.frame.columns:
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                r, p = stats.pointbiserialr(x, y)
                temp_r.append(r)
                temp_p.append(p)
            r_values.append(temp_r)
            p_values.append(temp_p)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        return df_r, df_p

    def kendall_tau_analysis(self):
        r_values = []
        p_values = []
        for column1 in self.frame.columns:
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                r, p = stats.kendalltau(x, y)
                temp_r.append(r)
                temp_p.append(p)
            r_values.append(temp_r)
            p_values.append(temp_p)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        return df_r, df_p

    def weighted_tau_analysis(self):
        r_values = []
        p_values = []
        for column1 in self.frame.columns:
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                r, p = stats.weightedtau(x, y)
                temp_r.append(r)
                temp_p.append(p)
            r_values.append(temp_r)
            p_values.append(temp_p)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        return df_r, df_p

    def lin_reg_analysis(self):
        slope_values = []
        inter_values = []
        r_values = []
        p_values = []
        std_err_values = []

        for column1 in self.frame.columns:
            temp_slope = []
            temp_inter = []
            temp_stderr = []
            temp_inter_stderr = []
            temp_r = []
            temp_p = []
            for column2 in self.frame.columns:
                x = self.frame[column1].values
                y = self.frame[column2].values
                res = stats.linregress(x, y)
                temp_slope.append(res.slope)
                temp_inter.append(res.intercept)
                temp_stderr.append(res.stderr)
                temp_r.append(res.rvalue)
                temp_p.append(res.pvalue)
            slope_values.append(temp_slope)
            inter_values.append(temp_inter)
            r_values.append(temp_stderr)
            p_values.append(temp_p)
            std_err_values.append(temp_r)
        r_values = np.array(r_values)
        p_values = np.array(p_values)
        slope_values = np.array(slope_values)
        inter_values = np.array(inter_values)
        std_err_values = np.array(std_err_values)
        df_r = pd.DataFrame(r_values, index = self.columns, columns = self.columns)
        df_p = pd.DataFrame(p_values, index = self.columns, columns = self.columns)
        df_slope = pd.DataFrame(slope_values, index = self.columns, columns = self.columns)
        df_inter = pd.DataFrame(inter_values, index = self.columns, columns = self.columns)
        df_stderr = pd.DataFrame(std_err_values, index = self.columns, columns = self.columns)
        return df_r, df_p, df_slope, df_inter, df_stderr
