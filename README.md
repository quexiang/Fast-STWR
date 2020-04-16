# Fast-STWR
News: 
1.Welcome to read and comment our article "A Spatiotemporal Weighted Regression Model (STWRv1.0) for Analyzing Local Non-stationarity in Space and Time",which can be accessed via the link below:https://doi.org/10.5194/gmd-2019-292 

2.We are now developing a network tool for Fast-STWR. 

Parallel Computing for Fast Spatiotemporal Weighted Regression

• F-STWR, a parallel computing method, is implemented in spatiotemporal weighted regression (STWR).

• A matrix splitting approach is developed for memory saving in STWR.

• F-STWR significantly improve the capability of processing large-scale spatiotemporal data.

To improve the efficiency of computing, we adopted a parallel computing method for STWR by employing the Message Passing Interface (MPI). A cache in the MPI processing approach was proposed for the calibration routine. We also designed a matrix splitting strategy to address the problem of memory insufficiency. We named the overall design as Fast STWR (F-STWR).We tested F-STWR in a High-Performance Computing (HPC) environment with a total number of 204611 observations from 19 years.The results show that F-STWR can significantly improve STWR’s capability of processing large-scale spatiotemporal data. 
