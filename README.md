# Fast-STWR

News: 
1.Welcome to read and comment our article "A Spatiotemporal Weighted Regression Model (STWRv1.0) for Analyzing Local Non-stationarity in Space and Time",which can be accessed via the link below:https://doi.org/10.5194/gmd-2019-292 

2.We are now developing a network tool for Fast-STWR. 

Parallel Computing for Fast Spatiotemporal Weighted Regression

• F-STWR, a parallel computing method, is implemented in spatiotemporal weighted regression (STWR).

• A matrix splitting approach is developed for memory saving in STWR.

• F-STWR significantly improve the capability of processing large-scale spatiotemporal data.

To improve the efficiency of computing, we adopted a parallel computing method for STWR by employing the Message Passing Interface (MPI). A cache in the MPI processing approach was proposed for the calibration routine. We also designed a matrix splitting strategy to address the problem of memory insufficiency. We named the overall design as Fast STWR (F-STWR).We tested F-STWR in a High-Performance Computing (HPC) environment with a total number of 204611 observations from 19 years.The results show that F-STWR can significantly improve STWR’s capability of processing large-scale spatiotemporal data. 
-------------------------------------------------------------------------------------------------------------------------------------------
Step 1:You should unzip the sphinx.zip and pytz.zip and put the directory to \dist\faststwr-mpi\ 

Step 2:You should unzip mkl_avx.zip mkl_avx2.zip mkl_avx512.zip mkl_core.zip mkl_def.zip mkl_intel_thread.zip mkl_mc.zip mkl_mc3.zip mkl_pgi_thread.zip  and  put all the *.dll files to the directory \dist\faststwr-mpi\
-------------------------------------------------------------------------------------------------------------------------------------------
