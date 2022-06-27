from read_data import *
from categorical_encoder import *
from feature_normalizer import *
from advanced_analysis_spatiotemporal import AdvancedAnalyser
from driver_exploration import *
from advanced_analysis_driver import *
import json


def get_processed_p_data(path):
    data = ReadData(path = path)
    data.read()
    df_orig = data.get_data()
    base_stats = DataStat(df_orig)
    base_stats.print_dimensions()
    categorical_columns = ['V1_Dir', 'V1_Action', 'V1_Lane_Nu', 'V1_Driver1',
                           'V2_Driver1', 'V2_Dir', 'V2_Action', 'V2_Lane_Nu',
                           'Total_Vehi', 'Fatalities', 'Injury_Typ', 'Injured', 'Property_D',
                           'Crash_Type']
    rest_columns = ['V1_Vehicle','V2_Driver_',
                    'Pedalcycli', 'Pedestrian', 'Motorcycli', 'V1_Type']
    df_cat = df_orig[categorical_columns]
    df_rest = df_orig[rest_columns]
    encoder = Encoder(df_cat, cat_columns = categorical_columns)
    encoded_data = encoder.get_onehot_encoded_data(columns = categorical_columns)
    print(encoded_data.head())
    encoded_data = pd.concat([df_cat, df_rest], axis=0)
    return encoded_data


def get_processed_st_data(path):
    data = ReadData(path = path)
    data.read()
    df_orig = data.get_data()
    base_stats = DataStat(df_orig)
    base_stats.print_dimensions()
    # print(base_stats.get_description())
    categorical_columns = ["Lighting", "Weather", "HWY_Factor", "Factors_Ro", "Crash_Seve", "Dir", "Primary_St",
                           "Agency",
                           "Secondary_", "Crash_Date"]
    encoder = Encoder(data=df_orig, cat_columns=categorical_columns)
    encoded_data = encoder.get_encoded_data()
    # print("Encoded Data: \n")
    # print(encoded_data.head(10))
    features_to_norm = ['X', 'Y', 'Distance', 'Accident_R', 'AREA', 'PERIMETER', 'WARD',
                        'ACRES', 'SQ_MILES', 'AREA_1', 'LEN']
    normalizer = FeatureNormalizer(data=encoded_data, data_desc=base_stats.get_description(),
                                   features_to_normalize=features_to_norm)
    normalized_data = normalizer.get_normalized_frame()
    # print("Normalized Data:")
    # print(normalized_data.head(10))
    return normalized_data


def get_advanced_analysis(data, measure_columns):
    analyser = AdvancedAnalyser(data, measure_columns)
    # chi_stats, chi_p = analyser.get_chi_stats()
    # t_stats, t_p = analyser.get_t_stats(col_a = 'PERIMETER', col_b = "AREA")
    # return chi_stats, chi_p, t_stats, t_p
    summary_stats = analyser.get_summary_measures()
    correlation_stats = analyser.get_correlation_stats()
    return summary_stats, correlation_stats


def do_correlation_analysis_driver(data):
    driver_analyser = AdvancedAnalyserDriver(data)
    print("Correlation Analysis Started...")
    pearson_r, pearson_p = driver_analyser.pearson_analysis()
    pearson_r.to_csv('pearson_r.csv')
    pearson_p.to_csv('pearson_p.csv')
    spearman_r, spearman_p = driver_analyser.spearman_analysis()
    spearman_r.to_csv('spearman_r.csv')
    spearman_p.to_csv('spearman_p.csv')
    pb_r, pb_p = driver_analyser.point_biserial_analysis()
    pb_r.to_csv('point_biserial_r.csv')
    pb_p.to_csv('point_biserial_p.csv')
    kt_r, kt_p = driver_analyser.kendall_tau_analysis()
    kt_r.to_csv('kendal_tau_r.csv')
    kt_p.to_csv('kendal_tau_p.csv')
    wt_r, wt_p = driver_analyser.weighted_tau_analysis()
    wt_r.to_csv('weighted_tau_r.csv')
    wt_p.to_csv('weighted_tau_p.csv')
    lr_r, lr_p, lr_slope, lr_inter, lr_stderr = driver_analyser.lin_reg_analysis()
    lr_r.to_csv('lin_reg_r.csv')
    lr_p.to_csv('lin_reg_p.csv')
    lr_slope.to_csv('lin_reg_slope.csv')
    lr_inter.to_csv('lin_reg_inter.csv')
    lr_stderr.to_csv('lin_reg_stderr.csv')
    print("Analysis Done!!")


def plot_histogram_driver(data, columns = None):
    plotter = HistPlotter(data)
    if columns:
        plotter.plot_for_columns(columns = columns)
    else:
        plotter.plot_for_all_columns()


def save_dataset(data, file_name):
    data.to_csv(file_name)
    print("Data Saved!!")


if __name__ == '__main__':
    # _______________________Spatio Temporal Data Analysis_________________________________
    # DATA_PATH = "spatiotemporal.csv"
    # processed_data = get_processed_st_data(DATA_PATH)
    # print(processed_data.head(10))
    # measure_column = ['X', 'Y', 'Distance', 'Accident_R', 'AREA', 'PERIMETER', 'WARD',
    #                             'ACRES', 'SQ_MILES', 'AREA_1', 'LEN']
    # summary_stat, correlation_stat = get_advanced_analysis(processed_data, measure_column)
    # print("_______________________________________________________________")
    # print(summary_stat)
    # print("_______________________________________________________________")
    # print(correlation_stat)

    # _________________________Praxeology Data Analysis____________________________
    # DATA_PATH_PRAXEOLOGY = "driver_processed.csv"
    # df_encoded_p = get_processed_p_data(DATA_PATH_PRAXEOLOGY)
    # df_explorer = DriverExplorer(df_encoded_p)
    # for column in ['V1_Dir', 'V1_Action', 'V1_Lane_Nu', 'V1_Driver1',
    #                        'V2_Driver1', 'V2_Dir', 'V2_Action', 'V2_Lane_Nu',
    #                        'Total_Vehi', 'Fatalities', 'Injury_Typ', 'Injured', 'Property_D',
    #                        'Crash_Type']:
    #       df_explorer.show_histogram(column)
    # df_explorer.show_heatmap()
    df_path = 'data/driver_readied.csv'
    dataclass = ReadData(df_path)
    df = dataclass.get_data()
    df.pop(list(df.columns)[0])
    print(df.head().columns)

    # _______________________Driver Data Advanced Correlation Analysis_____________________ #
    # do_correlation_analysis_driver(df)
    #________________________Driver Data Histogram Plots___________________________________ #
    plot_histogram_driver(df)



