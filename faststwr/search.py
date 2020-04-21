#Bandwidth optimization methods

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
from scipy import linalg
from copy import deepcopy
import copy
from collections import namedtuple
import time


def golden_section(a, c, delta, function, tol, max_iter, int_score=False):
    """
    Golden section search routine
    Method: p212, 9.6.4
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    a               : float
                      initial max search section value
    b               : float
                      initial min search section value
    delta           : float
                      constant used to determine width of search sections
    function        : function
                      obejective function to be evaluated at different section
                      values
    int_score       : boolean
                      False for float score, True for integer score
    tol             : float
                      tolerance used to determine convergence
    max_iter        : integer
                      maximum iterations if no convergence to tolerance

    Returns
    -------
    opt_val         : float
                      optimal value
    opt_score       : kernel
                      optimal score
    output          : list of tuples
                      searching history
    """
    b = a + delta * np.abs(c-a)
    d = c - delta * np.abs(c-a)
    score = 0.0
    diff = 1.0e9
    iters  = 0
    output = []
    dict = {}
    while np.abs(diff) > tol and iters < max_iter:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)
        
        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
        
        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c-a)
            #if int_score:
                #b = np.round(b)
        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c-a)
            #if int_score:
                #d = np.round(b)

        #if int_score:
        # opt_val = np.round(opt_val)
        output.append((opt_val, opt_score))
        diff = score_b - score_d
        score = opt_score
    return np.round(opt_val, 2), opt_score, output

def equal_interval(l_bound, u_bound, interval, function, int_score=False):
    """
    Interval search, using interval as stepsize

    Parameters
    ----------
    l_bound         : float
                      initial min search section value
    u_bound         : float
                      initial max search section value
    interval        : float
                      constant used to determine width of search sections
    function        : function
                      obejective function to be evaluated at different section
                      values
    int_score       : boolean
                      False for float score, True for integer score

    Returns
    -------
    opt_val         : float
                      optimal value
    opt_score       : kernel
                      optimal score
    output          : list of tuples
                      searching history
    """
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a,0)
        c = np.round(c,0)
        b = np.round(b,0)

    output = []

    score_a = function(a)
    score_c = function(c)

    output.append((a,score_a))
    output.append((c,score_c))

    if score_a < score_c:
        opt_val = a
        opt_score = score_a
    else:
        opt_val = c
        opt_score = score_c

    while b < c:
        score_b = function(b)

        output.append((b,score_b))

        if score_b < opt_score:
            opt_val = b
            opt_score = score_b
        b = b + interval

    return opt_val, opt_score, output

def equal_stwr_interval(l_bound, u_bound, interval, function, int_score=False):
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a,0)
        c = np.round(c,0)
        b = np.round(b,0)
    output = []   
    score_a = function(a)
    score_c = function(c)
    
#    while ((score_a < 0) and (a < c)):
#            a = a+ interval
#            score_a = function(a) 
#    while ((score_c < 0) and (c > a)):
#            c = c-interval
#            score_c = function(c)
#    b = a + interval
#    if int_score:
#        b = np.round(b,0)
    output.append((a,score_a))
    output.append((c,score_c))

    if score_a < score_c:
        opt_val = a
        opt_score = score_a
    else:
        opt_val = c
        opt_score = score_c

    while b < c:
        score_b = function(b)

        output.append((b,score_b))

        if (score_b < opt_score): #and (score_b > 0):
            opt_val = b
            opt_score = score_b
        b = b + interval

    return opt_val, opt_score, output

def multi_bw(init, y, X, n, k, family, tol, max_iter, rss_score,
        gwr_func, bw_func, sel_func, multi_bw_min, multi_bw_max):
    """
    Multiscale GWR bandwidth search procedure using iterative GAM backfitting
    """
    if init is None:
        bw = sel_func(bw_func(y, X))
        optim_model = gwr_func(y, X, bw)
    else:
        optim_model = gwr_func(y, X, init)
     
    S = optim_model.S
    err = optim_model.resid_response.reshape((-1,1))
    param = optim_model.params
    
    R = np.zeros((n,n,k))
    
    for j in range(k):
        for i in range(n):
            wi = optim_model.W[i].reshape(-1,1)
            xT = (X * wi).T
            P = linalg.solve(xT.dot(X), xT)
            R[i,:,j] = X[i,j]*P[j]

    XB = np.multiply(param, X)
    if rss_score:
        rss = np.sum((err)**2)
    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    VALs = []
    FUNCs = []
    
    try:
        from tqdm import tqdm #if they have it, let users have a progress bar
    except ImportError:
        def tqdm(x): #otherwise, just passthrough the range
            return x
    for iters in tqdm(range(1, max_iter+1)):
        new_XB = np.zeros_like(X)
        bws = []
        vals = []
        funcs = []
        current_partial_residuals = []
        params = np.zeros_like(X)
        f_XB = XB.copy()
        f_err = err.copy()
        
        for j in range(k):
            temp_y = XB[:,j].reshape((-1,1))
            temp_y = temp_y + err
            temp_X = X[:,j].reshape((-1,1))
            bw_class = bw_func(temp_y, temp_X)
            funcs.append(bw_class._functions)
            bw = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j])
            optim_model = gwr_func(temp_y, temp_X, bw)
            Aj = optim_model.S
            new_Rj = Aj - np.dot(Aj, S) + np.dot(Aj, R[:,:,j])
            S = S - R[:,:,j] + new_Rj
            R[:,:,j] = new_Rj
            
            err = optim_model.resid_response.reshape((-1,1))
            param = optim_model.params.reshape((-1,))

            new_XB[:,j] = optim_model.predy.reshape(-1)
            bws.append(copy.deepcopy(bw))
            params[:,j] = param
            vals.append(bw_class.bw[1])
            current_partial_residuals.append(err.copy())

        num = np.sum((new_XB - XB)**2)/n
        den = np.sum(np.sum(new_XB, axis=1)**2)
        score = (num/den)**0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1,1))
            new_rss = np.sum((y - predy)**2)
            score = np.abs((new_rss - rss)/new_rss)
            rss = new_rss
        scores.append(copy.deepcopy(score))
        delta = score
        BWs.append(copy.deepcopy(bws))
        VALs.append(copy.deepcopy(vals))
        FUNCs.append(copy.deepcopy(funcs))
        if delta < tol:
            break

    opt_bws = BWs[-1]
    return (opt_bws, np.array(BWs),
                          np.array(scores), params,
                          err, S, R)

def gg_stwr_section(a, c, delta, function, tol, max_iter, int_score=True):
    """
    Golden section search routine
    Method: p212, 9.6.4
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    a               : float
                      initial max search section value
    b               : float
                      initial min search section value
    delta           : float
                      constant used to determine width of search sections
    function        : function
                      obejective function to be evaluated at different section
                      values
    int_score       : boolean
                      False for float score, True for integer score
    tol             : float
                      tolerance used to determine convergence
    max_iter        : integer
                      maximum iterations if no convergence to tolerance

    Returns
    -------
    opt_val         : float
                      optimal value
    opt_score       : kernel
                      optimal score
    output          : list of tuples
                      searching history
    """
    b = a + delta * np.abs(c-a)
    d = c - delta * np.abs(c-a)
    score = 0.0
    diff = 1.0e9
    iters  = 0
    output = []
    dict = {} 
    opt_alpha = 0.0
    st_2 = time.clock()
    while np.abs(diff) > tol and iters < max_iter:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)
        
        if b in dict:
            score_b = dict[b]
        else:
            score_b,opt_alpha_b = function(b)
            dict[b] = score_b
        
        if d in dict:
            score_d = dict[d]
        else:
            score_d,opt_alpha_d = function(d)
            dict[d] = score_d

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            opt_alpha = opt_alpha_b
            c = d
            d = b
            b = a + delta * np.abs(c-a)
            #if int_score:
                #b = np.round(b)
        else:
            opt_val = d
            opt_score = score_d
            opt_alpha = opt_alpha_d
            a = b
            b = d
            d = c - delta * np.abs(c-a)
            #if int_score:
                #d = np.round(b)
        
        #if int_score:
        # opt_val = np.round(opt_val)
        output.append((opt_val, opt_score,opt_alpha))
        diff = score_b - score_d
        score = opt_score
        
        p2 = round((iters + 1) * 100 / max_iter)
        duration_tck = round(time.clock() - st_2, 2)
        remaining_tck = round(duration_tck * 100 / (0.01 + p2) - duration_tck, 2)
        print("每期搜索过程的进度:{0}%，已耗时:{1}s，预计剩余时间:{2}s\n".format(p2, duration_tck, remaining_tck), end="\r")
        time.sleep(0.01)
        
    return np.round(opt_val, 2), opt_score,opt_alpha,output