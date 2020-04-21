"""
Diagnostics for estimated gwr models
"""
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


import math
import numpy as np
from scipy import linalg
from spglm.family import Gaussian, Poisson, Binomial

def get_Stwr_CV(stwr):

    aa = stwr.resid_response.reshape((-1,1))/(1.0-stwr.influ)
    cv = np.sum(aa**2)/stwr.n
    return cv

def get_Stwr_AICc(stwr):
    """
    Get AICc value
    
    Gaussian: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    
    GWGLM: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)

    """
    if(stwr.tr_S >0):
        n = stwr.n
        k = stwr.tr_S 
        if isinstance(stwr.family, Gaussian):
            aicc = -2.0*stwr.llf + 2.0*n*(k + 1.0)/(n-k-2.0)
        elif isinstance(stwr.family, (Poisson, Binomial)):
            aicc = get_AIC(stwr) + 2.0 * k * (k+1.0) / (n - k - 1.0)
    else:
        aicc = np.inf
    if((stwr.R2<0.2) or (stwr.R2 > 1)or (aicc < 0)):
        aicc = np.inf
    return aicc

def get_AICc(gwr):
    """
    Get AICc value
    
    Gaussian: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    
    GWGLM: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)

    """
    n = gwr.n
    k = gwr.tr_S
    #sigma2 = gwr.sigma2
    if isinstance(gwr.family, Gaussian):
        aicc = -2.0*gwr.llf + 2.0*n*(k + 1.0)/(n-k-2.0) #equivalent to below but
        #can't control denominator of sigma without altering GLM familt code
        #aicc = n*np.log(sigma2) + n*np.log(2.0*np.pi) + n*(n+k)/(n-k-2.0)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aicc = get_AIC(gwr) + 2.0 * k * (k+1.0) / (n - k - 1.0) 
    return aicc

def get_AIC(gwr):
    """
    Get AIC calue

    Gaussian: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)

    GWGLM:  AIC(G)=D(G) + 2K(G), where D and K denote the deviance and the effective
    number of parameters in the model with bandwidth G, respectively.
    
    """   
    k = gwr.tr_S
    #deviance = -2*log-likelihood
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        aic = -2.0 * gwr.llf + 2.0 * (k+1)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aic = np.sum(gwr.family.resid_dev(y, mu)**2) + 2.0 * k
    return aic 

def get_BIC(gwr):
    """
    Get BIC value

    Gaussian: p61 (2.34), Fotheringham, Brunsdon and Charlton (2002)
    BIC = -2log(L)+klog(n)

    GWGLM: BIC = dev + tr_S * log(n)

    """
    n = gwr.n      # (scalar) number of observations
    k = gwr.tr_S  
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        bic = -2.0 * gwr.llf + (k+1) * np.log(n) 
    elif isinstance(gwr.family, (Poisson, Binomial)):
        bic = np.sum(gwr.family.resid_dev(y, mu)**2) + k * np.log(n)
    return bic


def get_CV(gwr):
    """
    Get CV value

    Gaussian only

    Methods: p60, (2.31) or p212 (9.4)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    Modification: sum of residual squared is divided by n according to GWR4 results

    """
    aa = gwr.resid_response.reshape((-1,1))/(1.0-gwr.influ)
    cv = np.sum(aa**2)/gwr.n
    return cv

def corr(cov):
    invsd = np.diag(1/np.sqrt(np.diag(cov)))
    cors = np.dot(np.dot(invsd, cov), invsd)
    return cors
