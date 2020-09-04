# Fast-STWR
**F**ast- **S**patiotemporal **W**eighted **R**egression (Fast-STWR)
=======================================
Fast-Spatiotemporal Weighted Regression

Parallel Computing for Fast Spatiotemporal Weighted Regression

• F-STWR, a parallel computing method, is implemented in spatiotemporal weighted regression (STWR).

• A matrix splitting approach is developed for memory saving in STWR.

• F-STWR significantly improve the capability of processing large-scale spatiotemporal data.

To improve the efficiency of computing, we adopted a parallel computing method for STWR by employing the Message Passing Interface (MPI). A cache in the MPI processing approach was proposed for the calibration routine. We also designed a matrix splitting strategy to address the problem of memory insufficiency. We named the overall design as Fast STWR (F-STWR).We tested F-STWR in a High-Performance Computing (HPC) environment with a total number of 204611 observations from 19 years.The results show that F-STWR can significantly improve STWR’s capability of processing large-scale spatiotemporal data. 

----------------------------------------------------------------------------------------------------------------------------------------
You can choose to use the released version of the Windows installer or execute python code to complete your spatiotemporal data analysis tasks.
If you excute the F-STWR_XXX.exe and it do not come out the summary file, please check that you can run the "mpiexec" CMD command in your windows system. You may need to install MPICH https://www.mpich.org/downloads/ and then add the directory of the installed MPICH "bin" folder to your system environment variables "PATH". And you may also make sure that the directory of F-STWR_XXX.exe is in the variables "PATH". 

----------------------------------------------------------------------------------------------------------------------------------------
Step 1:You should unzip the sphinx.zip and pytz.zip and put the directory to \dist\faststwr-mpi\ 

Step 2:You should unzip mkl_avx.zip mkl_avx2.zip mkl_avx512.zip mkl_core.zip mkl_def.zip mkl_intel_thread.zip mkl_mc.zip mkl_mc3.zip mkl_pgi_thread.zip  and  put all the *.dll files to the directory \dist\faststwr-mpi\

----------------------------------------------------------------------------------------------------------------------------------------
You can use less than 6 processors to run our FastSTWR,if you need more processors please contact us: xiangq@uidaho.edu or quexiang@fafu.edu.cn

----------------------------------------------------------------------------------------------------------------------------------------
  To use the FastSTWR, you should install the environment:
  
--------------------------------------------------------------------------------------------------------------------------------------
   FastSTWR-MPI is an implemention of STWR based on Message Passing Interface(MPI).
   To initial and excute the code, we need install MPICH in windows or openmpi in Linux 
   and the mpi4py is needed.
  
   Parameters
   --------------------  
   
   n: number of processors for model calibration of STWR
   Intervaldata : text file that record the list of time intervals and the first element in the list is zero.
                  It can be a text file like below:
                        0.000000000000000000e+00
                        1.000000000000000000e+02
                        1.000000000000000000e+02
                        1.000000000000000000e+02
                        1.000000000000000000e+02
                  5 time interval is recorded in the text file.
   out:  specify the directory and name where the resultfile will be output.
   schmethod: set the search method for calibration of STWR in MPI.
   criterion: set the criterion for model calibration.
   schtol: set the minimum convergence value.If the difference of two value is less than this value it will be regard as same.
   max_iter: set the maximum searching steps for model calibrition.
   eps: a small value to make sure that the matrix have non-singular solution.eps is zero mean and very small standar variance (1.0-7)
   family: specify the family object. underlying probability model; provides distribution-specific calculations 
   pysal: specify whether add constant to the matrix of independent variable.
   
   Examples
   --------
   
   (1) if you have the faststwr-mpi.py file:
   
   --------------------------------------------------------------------
   mpi_cmd = 'mpiexec'+' -n '
   +str(self.nproc)+' python '+ os.getcwd()+ '/mgwr/faststwr-mpi.py '
   +' -DataFile ' +dfnames
   +' -Intervaldata '+dftime
   +' -out '+resultfile 
   +' -schmethod '+self.search_method 
   +' -criterion '+self.criterion
   +' -schtol '+str(self.tol)
   +' -max_iter '+str(self.max_iter)
   +' -eps '+str(self.eps) 
   +' -family '+str(familytpye)
   +' --pysal' + ' --constant' 
   
   --------------------------------------------------------------------
   (2)if you do not have the faststwr-mpi.py file
   
   --------------------------------------------------------------------
   step1.You should find the faststwr-mpi.exe and set it to your working directory (you can use python command "os.getcwd()" to get the directory)
   step2.You can find the function "_mpi_spt_bw" of the class "Sel_Spt_BW" in the sel_bw file, and change the  Parameters above as you want.
   
   mpi_cmd = 'mpiexec'+' -n '+ str(self.nproc) 
   +' ' + os.getcwd() + '/fastmpiexc/dist/faststwr-mpi/' 
   + 'faststwr-mpi.exe' + ' -DataFile ' +dfnames+' -Intervaldata '
   +dftime+' -out '+resultfile +' -schmethod '
   +self.search_method +' -criterion '+self.criterion+' -schtol '
   +str(self.tol)+' -max_iter '+str(self.max_iter)+' -eps '
   +str(self.eps) +' -family '+str(familytpye)+' --pysal' + ' --constant'
   
   ------------------------------------------------------------------------------------------------------------------------------------- 
   How to Use FastSTWR
   
   ------------------------------------------------------------------------------------------------------------------------------------  
   ### fitting ###  
   
   stwr_selector_ = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,time_dif ,spherical = False)    
   optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search(nproc = 6) #you can change the nproc number  
   stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,time_dif,optsita,opt_gwr_bw0,tick_nums=opt_btticks,alpha =optalpha,spherical = False,recorded=1)
   stwr_results = stwr_model.fit()
   print(stwr_results.summary())
   
   ### prediction ### 
   stwr_scale = stwr_results.scale 
   stwr_residuals = stwr_results.resid_response
   
   ###predPointList is a list of coordinates of points to predict, PreX_list is a list of X at these coordinates. ###
   
   pred_stwr =  stwr_model.predict(predPointList,PreX_list,stwr_scale,stwr_residuals)
   pred_stwr_result =pred_stwr.predictions 
   
   -------------------------------------------------------------------------------------------------------------------------------------
   Thank you for your attention.If you find any bugs in codes, please don't hesitate to contact us.
   
   -------------------------------------------------------------------------------------------------------------------------------------
