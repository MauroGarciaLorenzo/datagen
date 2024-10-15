import os
from utils_pp_standalone import *

dir_name='ACOPF_standalone_seed16_nc3_ns100_20241014_162634_9593'
path='../results/'
path_results = os.path.join(path, dir_name)

results_dataframes, csv_files= open_csv(path_results)

perc_stability(results_dataframes['case_df_op'])


