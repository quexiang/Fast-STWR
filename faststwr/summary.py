__author__ = "STWR,F-STWR is XiangQue xiangq@uidaho.edu and GWR,MGWR is Taylor Oshan Tayoshan@gmail.com"

"""
  You can use less than 6 processors to run our FastSTWR,if you need more processors please contact us: xiangq@uidaho.edu or quexiang@fafu.edu.cn
  ---------------------------------------------------------------------------------------------------------------------------------------------------- 
  To use the FastSTWR, you should install the environment:
  ---------------------------------------------------------------------------------------------------------------------------------------------------- 
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
   ----------------------------------------------------------------------------------------------------------------------------------------------------  
   How to Use FastSTWR
   ----------------------------------------------------------------------------------------------------------------------------------------------------
   ### fitting ###
   ### Please refer to the STWR model###
   
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
   
   ----------------------------------------------------------------------------------------------------------------------------------------------------
   Thank you for your attention.If you find any bug of the program, please contact us!
   ----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM
from .diagnostics import get_AICc

def summaryModel(self):
    summary = '=' * 75 + '\n'
    summary += "%-54s %20s\n" % ('Model type', self.family.__class__.__name__)
    summary += "%-60s %14d\n" % ('Number of observations:', self.n)
    summary += "%-60s %14d\n\n" % ('Number of covariates:', self.k)
    return summary

def summaryGLM(self):
    
    XNames = ["X"+str(i) for i in range(self.k)]
    glm_rslt = GLM(self.model.y,self.model.X,constant=False,family=self.family).fit()

    summary = "%s\n" %('Global Regression Results')
    summary += '-' * 75 + '\n'
    
    if isinstance(self.family, Gaussian):
        summary += "%-62s %12.3f\n" %  ('Residual sum of squares:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" %  ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" %  ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" %  ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" %  ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" %  ('R2:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. R2:', glm_rslt.adj_D2)
    else:
        summary += "%-62s %12.3f\n" %  ('Deviance:', glm_rslt.deviance)
        summary += "%-62s %12.3f\n" %  ('Log-likelihood:', glm_rslt.llf)
        summary += "%-62s %12.3f\n" %  ('AIC:', glm_rslt.aic)
        summary += "%-62s %12.3f\n" %  ('AICc:', get_AICc(glm_rslt))
        summary += "%-62s %12.3f\n" %  ('BIC:', glm_rslt.bic)
        summary += "%-62s %12.3f\n" %  ('Percent deviance explained:', glm_rslt.D2)
        summary += "%-62s %12.3f\n\n" % ('Adj. percent deviance explained:', glm_rslt.adj_D2)
    
    summary += "%-31s %10s %10s %10s %10s\n" % ('Variable', 'Est.', 'SE' ,'t(Est/SE)', 'p-value')
    summary += "%-31s %10s %10s %10s %10s\n" % ('-'*31, '-'*10 ,'-'*10, '-'*10,'-'*10)
    for i in range(self.k):
        summary += "%-31s %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], glm_rslt.params[i], glm_rslt.bse[i], glm_rslt.tvalues[i], glm_rslt.pvalues[i])
    summary += "\n"
    return summary

def summaryGWR(self):
    XNames = ["X"+str(i) for i in range(self.k)]
    
    summary = "%s\n" %('Geographically Weighted Regression (GWR) Results')
    summary += '-' * 75 + '\n'

    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + self.model.kernel)

    summary += "%-62s %12.3f\n" % ('Bandwidth used:', self.model.bw)

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'
    
    if isinstance(self.family, Gaussian):
        
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.resid_ss)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-62s %12.3f\n" % ('R2:', self.R2)
    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        #summary += "%-60s %12.6f\n" % ('Percent deviance explained:', 0)


    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', self.adj_alpha[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', self.critical_tval(self.adj_alpha[1]))

    summary += "\n%s\n" % ('Summary Statistics For GWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean' ,'STD', 'Min' ,'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-'*20, '-'*10 ,'-'*10, '-'*10 ,'-'*10, '-'*10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], np.mean(self.params[:,i]) ,np.std(self.params[:,i]),np.min(self.params[:,i]) ,np.median(self.params[:,i]), np.max(self.params[:,i]))

    summary += '=' * 75 + '\n'

    return summary



def summaryMGWR(self):
    
    XNames = ["X"+str(i) for i in range(self.k)]
    
    summary = ''
    summary += "%s\n" %('Multi-Scale Geographically Weighted Regression (MGWR) Results')
    summary += '-' * 75 + '\n'
    
    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + self.model.kernel)

    summary += "%-54s %20s\n" % ('Criterion for optimal bandwidth:', self.model.selector.criterion)

    if self.model.selector.rss_score:
        summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'RSS')
    else:
        summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'Smoothing f')

    summary += "%-54s %20s\n\n" % ('Termination criterion for MGWR:', self.model.selector.tol_multi)

    summary += "%s\n" %('MGWR bandwidths')
    summary += '-' * 75 + '\n'
    summary += "%-15s %14s %10s %16s %16s\n" % ('Variable', 'Bandwidth', 'ENP_j','Adj t-val(95%)','Adj alpha(95%)')
    for j in range(self.k):
        summary += "%-14s %15.3f %10.3f %16.3f %16.3f\n" % (XNames[j], self.model.bw[j], self.ENP_j[j],self.critical_tval()[j],self.adj_alpha_j[j,1])

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'
    
    summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.resid_ss)
    summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
    summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
    
    summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
    summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
    summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
    summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
    summary += "%-62s %12.3f\n" % ('BIC:', self.bic)

    summary += "\n%s\n" % ('Summary Statistics For MGWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean' ,'STD', 'Min' ,'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-'*20, '-'*10 ,'-'*10, '-'*10 ,'-'*10, '-'*10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], np.mean(self.params[:,i]) ,np.std(self.params[:,i]),np.min(self.params[:,i]) ,np.median(self.params[:,i]), np.max(self.params[:,i]))

    summary += '=' * 75 + '\n'
    return summary

def summarySTWR(self):
    XNames = ["X"+str(i) for i in range(self.k)]
    
    summary = "%s\n" %('Fast Spatiotemporal Weighted Regression (F-STWR) Results')
    summary += '-' * 75 + '\n'

    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + self.model.kernel)

    summary += "%-62s %12.3f\n" % ('Model sita used:', self.model.sita)
    
    summary += "%-62s %12.3f\n" % ('Model alpha used:', self.model.alpha)
        
    summary += "%-62s %12.3f\n" % ('Init Bandwidth used:', self.model.gwr_bw0+1)
    
    summary += "%-62s %12.3f\n" % ('Model Ticktimes used:',  self.model.tick_nums)
    
    summary += "%-62s %12.3f\n" % ('Model Ticktimes Intervels:',  np.sum(self.model.tick_times_intervel))

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'
    
    if isinstance(self.family, Gaussian):
        
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.resid_ss)
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        summary += "%-62s %12.3f\n" % ('R2:', self.R2)
    else:
        summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df_model)
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
        summary += "%-62s %12.3f\n" % ('BIC:', self.bic)
        #summary += "%-60s %12.6f\n" % ('Percent deviance explained:', 0)


    summary += "%-62s %12.3f\n" % ('Adj. alpha (95%):', self.adj_alpha[1])
    summary += "%-62s %12.3f\n" % ('Adj. critical t value (95%):', self.critical_tval(self.adj_alpha[1]))

    summary += "\n%s\n" % ('Summary Statistics For STWR Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean' ,'STD', 'Min' ,'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-'*20, '-'*10 ,'-'*10, '-'*10 ,'-'*10, '-'*10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], np.mean(self.params[:,i]) ,np.std(self.params[:,i]),np.min(self.params[:,i]) ,np.median(self.params[:,i]), np.max(self.params[:,i]))

    summary += '=' * 75 + '\n'

    return summary



