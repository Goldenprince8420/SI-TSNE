from scipy import stats


class AdvancedAnalyser:
    def __init__(self, data, measure_columns):
        super(AdvancedAnalyser, self).__init__()
        self.frame_orig = data
        self.frame = self.frame_orig[measure_columns]
        self.chi_stat = None
        self.chi_p_values = None
        self.t_stat = None
        self.t_p_values = None
        self.skewness = None
        self.kurtosis = None
        self.hmean = None
        self.gmean = None
        self.kstat = None
        self.kstatvar = None
        self.tmean = None
        self.tstd = None
        self.tmin = None
        self.tmax = None
        self.tsem = None
        self.variation = None
        self.gstd = None
        self.iqr = None
        self.sem = None
        self.bayes_mvs = None
        self.mvsdist = None
        self.diff_entropy = None
        self.median_abs_deviation = None
        self.summary_measure_stats = {}
        self.correlation_measure_stats = {}
        self.pearson_coeff = {}
        self.pearson_p_value = {}

    def do_t_test(self, col_a, col_b):
        a = self.frame[col_a].values
        b = self.frame[col_b].values
        self.t_stat, self.t_p_values = stats.ttest_ind(a, b)

    def get_t_stats(self, col_a, col_b):
        self.do_t_test(col_a, col_b)
        return self.t_stat, self.t_p_values

    def central_measure(self):
        self.chi_stat, self.chi_p_values = stats.chisquare(self.frame)
        self.skewness = stats.skew(self.frame)
        self.kurtosis = stats.kurtosis(self.frame)
        # self.hmean = stats.hmean(self.frame)
        self.gmean = stats.gmean(self.frame)
        self.kstat = stats.kstat(self.frame)
        self.kstatvar = stats.kstatvar(self.frame)
        self.tmean = stats.tmean(self.frame)
        self.tstd = stats.tstd(self.frame)
        self.tmin = stats.tmin(self.frame)
        self.tmax = stats.tmax(self.frame)
        self.tsem = stats.tsem(self.frame)
        self.variation = stats.variation(self.frame)
        # self.gstd = stats.gstd(self.frame)
        self.iqr = stats.iqr(self.frame)
        self.sem = stats.sem(self.frame)
        self.bayes_mvs = stats.bayes_mvs(self.frame)
        self.mvsdist = stats.mvsdist(self.frame)
        self.diff_entropy = stats.differential_entropy(self.frame)
        self.median_abs_deviation = stats.median_abs_deviation(self.frame)
        self.summary_measure_stats['chi_stat'] = self.chi_stat
        self.summary_measure_stats['chi_p_values'] = self.chi_p_values
        self.summary_measure_stats['skewness'] = self.skewness
        self.summary_measure_stats['kurtosis'] = self.kurtosis
        # self.summary_measure_stats['hmean'] = self.hmean
        self.summary_measure_stats['gmean'] = self.gmean
        self.summary_measure_stats['kstat'] = self.kstat
        self.summary_measure_stats['kstatvar'] = self.kstatvar
        self.summary_measure_stats['tmean'] = self.tmean
        self.summary_measure_stats['tstd'] = self.tstd
        self.summary_measure_stats['tmin'] = self.tmin
        self.summary_measure_stats['tmax'] = self.tmax
        self.summary_measure_stats['tsem'] = self.tsem
        self.summary_measure_stats['variation'] = self.variation
        # self.summary_measure_stats['gstd'] = self.gstd
        self.summary_measure_stats['iqr'] = self.iqr
        self.summary_measure_stats['sem'] = self.sem
        self.summary_measure_stats['variation'] = self.variation
        self.summary_measure_stats['gstd'] = self.gstd
        self.summary_measure_stats['iqr'] = self.iqr
        self.summary_measure_stats['bayes_mvs'] = self.bayes_mvs
        self.summary_measure_stats['mvsdist'] = self.mvsdist
        self.summary_measure_stats['diff_entropy'] = self.diff_entropy
        self.summary_measure_stats['median_abs_deviation'] = self.median_abs_deviation

    def get_summary_measures(self):
        self.central_measure()
        return self.summary_measure_stats

    def correlations_stat(self):
        # ['X', 'Y', 'Distance', 'Accident_R', 'AREA', 'PERIMETER', 'WARD',
        #                         'ACRES', 'SQ_MILES', 'AREA_1', 'LEN']
        self.pearson_coeff['X_Y'], self.pearson_p_value['X_Y'] = \
            stats.pearsonr(self.frame['X'], self.frame['Y'])
        self.pearson_coeff['Accident_R_AREA'], self.pearson_p_value['Accident_R_AREA'] = \
            stats.pearsonr(self.frame['Accident_R'], self.frame['AREA'])
        self.pearson_coeff['Accident_R_WARD'], self.pearson_p_value['X_Y'] = \
            stats.pearsonr(self.frame['Accident_R'], self.frame['WARD'])
        self.pearson_coeff['Distance_AREA'], self.pearson_p_value['Distance_AREA'] = \
            stats.pearsonr(self.frame['Distance'], self.frame['AREA'])
        self.pearson_coeff['AREA_PERIMETER'], self.pearson_p_value['AREA_PERIMETER'] = \
            stats.pearsonr(self.frame['AREA'], self.frame['PERIMETER'])
        self.pearson_coeff['ACRES_DISTANCE'], self.pearson_p_value['ACRES_DISTANCE'] = \
            stats.pearsonr(self.frame['ACRES'], self.frame['Distance'])
        self.pearson_coeff['SQ_MILES_ACRES'], self.pearson_p_value['SQ_MILES_ACRES'] = \
            stats.pearsonr(self.frame['SQ_MILES'], self.frame['ACRES'])
        self.pearson_coeff['Accident_R_SQ_MILES'], self.pearson_p_value['Accident_R_SQ_MILES'] = \
            stats.pearsonr(self.frame['Accident_R'], self.frame['SQ_MILES'])
        self.correlation_measure_stats["Pearson_Coefficient"] = self.pearson_coeff
        self.correlation_measure_stats["Pearson_P_values"] = self.pearson_p_value

    def get_correlation_stats(self):
        self.correlations_stat()
        return self.correlation_measure_stats


        


