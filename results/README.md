# Summary about results files
## Folders
Each folder collects the .csv files with the results obtained trough an execution of either ACOPF_standalone.py or run_datagen_ACOPF.py codes. 
The name of the folder indicates:
- which code originates the files: ACOPF_standalone / datagen_ACOPF
- (eventually) number of converter interfaced generators in the system: NCIGx (as i'm testing what happened with a different number of converters)
- the seed number used when executing the code: seedx
- the number of sampled points in the space: nsx
- the number of cases = number of different scenarios related to converter power injection: ncx
- timestamp of when the code is executed
- random number

Each folder contains:
- case_df_computing_times.csv: with the computing times of each data set instance
- dims_df.csv: for each data set instance, the sampled quantities in the multidimensional space
- cases_df.csv: for each data set instance, all the sampled quantities, not only of the dimensions but also of all the variables 
- case_df_real.csv, case_df_image.csv, case_df_freq.csv, case_df_damping.csv: real parts, imaginary parts, frequency and damping of the eigenvalues obtained for each data set instance.

- 
