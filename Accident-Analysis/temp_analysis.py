import matplotlib.pyplot as plt
from main import *
import seaborn as sns
from scipy.stats import describe, skew
from driver_exploration import *
from scipy import stats


# DATA_FILE = "spatiotemporal.csv"
# dataset = get_processed_data(DATA_FILE)
#
# area_data = dataset['AREA']
#
# subplot_index = 241
# plt.figure(figsize = (25, 10))
# for column in ['AREA', 'X', 'Y', 'PERIMETER', 'ACRES', 'SQ_MILES', 'AREA_1', 'LEN']:
#     plt.subplot(subplot_index)
#     sns.histplot(dataset[column])
#     subplot_index += 1
# # plt.show()
#
# sns.heatmap(dataset.corr())
# # plt.show()
#
# dataset_desc = describe(dataset[['AREA', 'X', 'Y', 'PERIMETER', 'ACRES', 'SQ_MILES', 'AREA_1', 'LEN']])
# data_skewness = skew(dataset)
# data_kurtosis = dataset_desc.kurtosis
# print(data_skewness)
# # print(dataset_desc.kurtosis)

DATA_PATH_PRAXEOLOGY = "data/driver_processed.csv"
df_encoded_p = get_processed_p_data(DATA_PATH_PRAXEOLOGY)
df_explorer = DriverExplorer(df_encoded_p)
print(list(df_encoded_p.columns))

yk = df_encoded_p['Crash_Type'].values
xk = np.linspace(1, 12720, 25440)
print(yk.shape)
print(xk.shape)
custm_crash = stats.rv_discrete(name='custm_crash', values=(xk, yk))
