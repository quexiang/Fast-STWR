# STWR and GWR Bandwidth selection class
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

import spreg.user_output as USER
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import minimize_scalar
from spglm.family import Gaussian, Poisson, Binomial
from spglm.iwls import iwls,_compute_betas_gwr
from .kernels import *
from .gwr import GWR,STWR
#from .FastSTWR import FastSTWR
from .search import golden_section, equal_interval, multi_bw,equal_stwr_interval,gg_stwr_section
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV,get_Stwr_AICc,get_Stwr_CV
from functools import partial
from math import atan,tan,ceil
import sys
import time
import subprocess
import os
import faststwr
from tempfile import NamedTemporaryFile
from spglm.utils import cache_readonly
import pickle
from spglm.glm import GLM, GLMResults

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QProgressBar


kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6:adapt_exp,7:fix_spt_bisquare,8:adapt_spt_bisquare}
getDiag = {'AICc': get_AICc,'AIC':get_AIC, 'BIC': get_BIC, 'CV': get_CV,'Stwr_AICc': get_Stwr_AICc,'STWR_CV':get_Stwr_CV}

class Sel_BW(object):
    """
    This version you can use maximum 6 cores to excute the MPI based 
    
    you can set 
    
    search 
    
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).

    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel         : string
                     type of kernel used and wether fixed or adaptive
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search_method  : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval increment used in interval search
    tol            : float
                     tolerance used to determine convergence
    max_iter       : integer
                     max interations if no convergence to tol
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    dmat          : array
                    n*n, distance matrix between calibration locations used
                    to compute weight matrix
                        
    sorted_dmat   : array
                    n*n, sorted distance matrix between calibration locations used
                    to compute weight matrix. Will be None for fixed bandwidths
        
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
    search_params : dict
                    stores search arguments
    int_score     : boolan
                    True if adaptive bandwidth is being used and bandwdith
                    selection should be discrete. False
                    if fixed bandwidth is being used and bandwidth does not have
                    to be discrete.
    bw            : scalar or array-like
                    Derived optimal bandwidth(s). Will be a scalar for GWR
                    (multi=False) and a list of scalars for MGWR (multi=True)
                    with one bandwidth for each covariate.
    S             : array
                    n*n, hat matrix derived from the iterative backfitting
                    algorthim for MGWR during bandwidth selection
    R             : array
                    n*n*k, partial hat matrices derived from the iterative
                    backfitting algoruthm for MGWR during bandwidth selection.
                    There is one n*n matrix for each of the k covariates.
    params        : array
                    n*k, calibrated parameter estimates for MGWR based on the
                    iterative backfitting algorithm - computed and saved here to
                    avoid having to do it again in the MGWR object.

    Examples
    --------

    >>> import libpysal as ps
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    
    Golden section search AICc - adaptive bisquare

    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print(bw)
    93.0

    Golden section search AIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
    >>> print(bw)
    50.0

    Golden section search BIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
    >>> print(bw)
    62.0

    Golden section search CV - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
    >>> print(bw)
    68.0

    Interval AICc - fixed bisquare

    >>> sel = Sel_BW(coords, y, X, fixed=True)
    >>> bw = sel.search(search_method='interval', bw_min=211001.0, bw_max=211035.0, interval=2)
    >>> print(bw)
    211025.0

    """
    def __init__(self, coords, y, X_loc, X_glob=None, family=Gaussian(),
            offset=None, kernel='bisquare', fixed=False, multi=False,
            constant=True, spherical=False):
        self.coords = coords
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        if offset is None:
          self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self._build_dMat()
        self.search_params = {}

    def search(self, search_method='golden_section', criterion='AICc',
            bw_min=None, bw_max=None, interval=0.0, tol=1.0e-6, max_iter=200,
            init_multi=None, tol_multi=1.0e-5, rss_score=False,
            max_iter_multi=200, multi_bw_min=[None], multi_bw_max=[None]):
        """
        Method to select one unique bandwidth for a gwr model or a
        bandwidth vector for a mgwr model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search_method  : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        multi_bw_min   : list 
                         min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept
        multi_bw_max   : list
                         max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol
        init_multi     : float
                         None (default) to initialize MGWR with a bandwidth
                         derived from GWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.
        tol_multi      : convergence tolerence for the multiple bandwidth
                         backfitting algorithm; a larger tolerance may stop the
                         algorith faster though it may result in a less optimal
                         model
        max_iter_multi : max iterations if no convergence to tol for multiple
                         bandwidth backfittign algorithm
        rss_score      : True to use the residual sum of sqaures to evaluate
                         each iteration of the multiple bandwidth backfitting
                         routine and False to use a smooth function; default is
                         False

        Returns
        -------
        bw             : scalar or array
                         optimal bandwidth value or values; returns scalar for
                         multi=False and array for multi=True; ordering of bandwidths
                         matches the ordering of the covariates (columns) of the
                         designs matrix, X
        """
        k = self.X_loc.shape[1]
        if self.constant: #k is the number of covariates
            k +=1
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        
        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min*k
        else:
            raise AttributeError("multi_bw_min must be either a list containing"
            " a single entry or a list containing an entry for each of k"
            " covariates including the intercept")
        
        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max*k
        else:
            raise AttributeError("multi_bw_max must be either a list containing"
            " a single entry or a list containing an entry for each of k"
            " covariates including the intercept")
        
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_multi = init_multi
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params['search_method'] = search_method
        self.search_params['criterion'] = criterion
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        if self.fixed:
            if self.kernel == 'gaussian':
                ktype = 1
            elif self.kernel == 'bisquare':
                ktype = 3
            elif self.kernel == 'exponential':
                ktype = 5
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)
        else:
            if self.kernel == 'gaussian':
              ktype = 2
            elif self.kernel == 'bisquare':
                ktype = 4
            elif self.kernel == 'exponential':
                ktype = 6
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)

        if ktype % 2 == 0:
            int_score = True
        else:
            int_score = False
        self.int_score = int_score 

        if self.multi:
            self._mbw()
            self.params = self.bw[3] 
            self.S = self.bw[-2] 
            self.R = self.bw[-1] 
        else:
            self._bw()

        return self.bw[0]
    
    def _build_dMat(self):
        if self.fixed:
            self.dmat = cdist(self.coords,self.coords,self.spherical)
            self.sorted_dmat = None
        else:
            self.dmat = cdist(self.coords,self.coords,self.spherical)
            self.sorted_dmat = np.sort(self.dmat)
    def _bw(self):

        gwr_func = lambda bw: getDiag[self.criterion](GWR(self.coords, self.y, 
            self.X_loc, bw, family=self.family, kernel=self.kernel,
            fixed=self.fixed, constant=self.constant,
            dmat=self.dmat,sorted_dmat=self.sorted_dmat).fit(searching = True))
        self._optimized_function = gwr_func
        if self.search_method == 'golden_section':
            a,c = self._init_section(self.X_glob, self.X_loc, self.coords,
                    self.constant)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                    self.max_iter, self.int_score)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                    gwr_func, self.int_score)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(self.X_glob, self.X_loc,
                    self.coords, self.constant)
            if self.bw_min == self.bw_max:
                raise Exception('Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.')
            self._optimize_result = minimize_scalar(gwr_func, bounds=(self.bw_min,
                self.bw_max), method='bounded')
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError('Unsupported computational search method ',
                    self.search_method)
    def _mbw(self):
        y = self.y
        if self.constant:
            X = USER.check_constant(self.X_loc)
        else:
            X = self.X_loc
        n, k = X.shape
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
        coords = self.coords
        search_method = self.search_method
        criterion = self.criterion
        bw_min = self.bw_min
        bw_max = self.bw_max
        multi_bw_min = self.multi_bw_min
        multi_bw_max = self.multi_bw_max
        interval = self.interval
        tol = self.tol
        max_iter = self.max_iter
        def gwr_func(y,X,bw):
            return GWR(coords, y,X,bw,family=family, kernel=kernel, fixed=fixed,
                    offset=offset, constant=False).fit()
        def bw_func(y,X):
            return Sel_BW(coords, y,X,X_glob=[], family=family, kernel=kernel,
                    fixed=fixed, offset=offset, constant=False)
        def sel_func(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(search_method=search_method, criterion=criterion,
                    bw_min=bw_min, bw_max=bw_max, interval=interval, tol=tol, max_iter=max_iter)
        self.bw = multi_bw(self.init_multi, y, X, n, k, family,
                self.tol_multi, self.max_iter_multi, self.rss_score, gwr_func,
                bw_func, sel_func, multi_bw_min, multi_bw_max)

    def _init_section(self, X_glob, X_loc, coords, constant):
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        if constant:
            n_vars = n_glob + n_loc + 1
        else:
            n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]
        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            sq_dists = pdist(coords)
            a = np.min(sq_dists)/2.0
            c = np.max(sq_dists)*2.0
        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None:
            c = self.bw_max      
        return a, c
    
    
class Sel_Spt_BW(object):    
    """
    Select bandwidth for spatiotemporal kernel

    Parameters
    ----------
    (self, coordslist,y_list,X_list, 
                 tick_times_intervel,
                 dspal_mat_list=None,sorted_dspal_list=None,d_tmp_list=None,dspmat = None,dtmat=None,
                 family=Gaussian(),
                 offset=None, kernel='spt_bisquare', fixed=False, multi=False,eps=1.0000001,max_cal_tol =100000000,
                 constant=True, spherical=False):
    
    coordslist: list of coord tuples
                   (x,y) of points used in bandwidth selection    
    y_list:     list of array
                   each array is n*1, dependent variable at a certain time stage.
                   
    X_list:     list of array
                   each array is independent at a certain time stage.
    tick_times_intervel:
    dspal_mat_list:
    sorted_dspal_list:
    d_tmp_list:
    dspmat:
    dtmat:     
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    kernel         : string
                     kernel function: 'spt_gaussian', 'spt_bisquare', 'spt_exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    eps:            a small value to make sure that the matrix have non-singular solution.eps is 1.0-7 mean 0 and very small standar variance (1.0-7)
    
    max_cal_tol:    maximun value for separate the matrix into smaller ones. Default size is 100000000.
                    Used for model calibration when the  memory usage is limitted.
    
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).

    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel         : string
                     type of kernel used and wether fixed or adaptive
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search_method  : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval increment used in interval search
    tol            : float
                     tolerance used to determine convergence
    max_iter       : integer
                     max interations if no convergence to tol
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    dmat          : array
                    n*n, distance matrix between calibration locations used
                    to compute weight matrix
                        
    sorted_dmat   : array
                    n*n, sorted distance matrix between calibration locations used
                    to compute weight matrix. Will be None for fixed bandwidths
        
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
    search_params : dict
                    stores search arguments
    int_score     : boolan
                    True if adaptive bandwidth is being used and bandwdith
                    selection should be discrete. False
                    if fixed bandwidth is being used and bandwidth does not have
                    to be discrete.
    bw            : scalar or array-like
                    Derived optimal bandwidth(s). Will be a scalar for GWR
                    (multi=False) and a list of scalars for MGWR (multi=True)
                    with one bandwidth for each covariate.
    S             : array
                    n*n, hat matrix derived from the iterative backfitting
                    algorthim for MGWR during bandwidth selection
    R             : array
                    n*n*k, partial hat matrices derived from the iterative
                    backfitting algoruthm for MGWR during bandwidth selection.
                    There is one n*n matrix for each of the k covariates.
    params        : array
                    n*k, calibrated parameter estimates for MGWR based on the
                    iterative backfitting algorithm - computed and saved here to
                    avoid having to do it again in the MGWR object.

    Examples
    --------
    """
    def __init__(self, coordslist,y_list,X_list, tick_times_intervel,
                 dspal_mat_list=None,sorted_dspal_list=None,
                 d_tmp_list=None,dspmat = None,dtmat=None,family=Gaussian(),
                 offset=None, kernel='spt_bisquare', fixed=False, multi=False,eps=1.0000001,
                 max_cal_tol =100000000,constant=True, spherical=False):
        self.n_tick_nums = len(coordslist)
        self.coordslist = coordslist
        self.tick_timesIntls = tick_times_intervel
        self.y_list =y_list
        self.X_list = X_list
        self.dspal_mat_list =dspal_mat_list
        self.sorted_dspal_list = sorted_dspal_list
        self.d_tmp_list = d_tmp_list
        self.dspmat = dspmat
        self.dtmat = dtmat
        self.eps = eps
        self.lg_bw_sch_iter = 10
        self.alpha_sch_times = 25
        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self.max_cal_tol = max_cal_tol
        self.cur_len = len(coordslist[-1])
        self.mlist = [self.cur_len]
        self.cal_len_tol = len(coordslist[-1])
        self.separts = None

    def search(self, search_method='interval',#'golden_section',
               criterion= 'STWR_CV',#'Stwr_AICc',,#'Stwr_AICc',
            sita_min=-np.pi/2, sita_max=np.pi/2, interval=np.pi/200,Intls =1, tol=1.0e-6, max_iter=200,nproc=1,progress = None):
        self.search_method = search_method
        self.criterion = criterion
        self.sita_min = sita_min
        self.sita_max = sita_max  
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter        
        self.nproc = nproc 
        self.progress = progress
        
        
        for i in range(self.n_tick_nums-1):
            len_m_tick = len(self.coordslist[-(2+i)])
            self.mlist.append(len_m_tick)
            self.cal_len_tol += len_m_tick
        if(self.cur_len*self.cal_len_tol> self.max_cal_tol):
             max_tick = ceil(self.max_cal_tol*1.0/self.cal_len_tol)
             self.separts = ceil(self.cur_len*1.0/max_tick)
             if self.nproc < self.separts:
                 sys.exit("init search Error,Input Data is out of memory,Please set the nproc equal to or greater than self.separts%d\n",self.separts)    
        self._opt_alpha,self._opt_bsita,self._opt_btticks,self._opt_bw0 = self._mpi_spt_bw()
        
        return self._opt_alpha,self._opt_bsita,self._opt_btticks,self._opt_bw0

    def _mpi_spt_bw(self):
            datafiles = []
            for itick in range( self.n_tick_nums):
                 data_tick = np.hstack([np.array(self.coordslist[itick]),self.y_list[itick],self.X_list[itick]])
                 df = os.getcwd()+"/datafile{:d}.txt".format(itick)
                 np.savetxt(df, data_tick, delimiter=',',comments='')
                 datafiles.append(df)
            dfnames = os.getcwd()+"/my_mpitmp.txt"
            with open(dfnames,'w',encoding = 'utf-8') as f:
                 for it in datafiles:
                              f.write(it+"\n")
            
            timeIntervalData = np.asarray(self.tick_timesIntls) 
            dftime = os.getcwd()+"/dftime.txt"
            np.savetxt(dftime, timeIntervalData, delimiter=',',comments='')

            resultfile = os.getcwd()+"/mpirunresults.txt"
            familytpye = 0
            if isinstance(self.family, Gaussian): 
                                    familytpye = 0
            elif isinstance(self.family, (Poisson, Binomial)):
                                    familytpye = 1 

            #openmpi 4.0.2  command mpiexec -np/ --np / -c
            #mpi_cmd = 'mpiexec'+' -np '+str(self.nproc)+' python '+ os.getcwd()+ '/stwr/faststwr-mpi.py '+' -DataFile ' +dfnames+' -Intervaldata '+dftime+' -out '+resultfile +' -schmethod '+self.search_method +' -criterion '+self.criterion+' -schtol '+str(self.tol)+' -max_iter '+str(self.max_iter)+' -eps '+str(self.eps) +' -family '+str(familytpye)+' --pysal' + ' --constant'
            #mpich command mpiexec -n            
            #mpi_cmd = 'mpiexec'+' -n '+str(self.nproc)+' python '+ os.getcwd()+ '/stwr/faststwr-mpi.py '+' -DataFile ' +dfnames+' -Intervaldata '+dftime+' -out '+resultfile +' -schmethod '+self.search_method +' -criterion '+self.criterion+' -schtol '+str(self.tol)+' -max_iter '+str(self.max_iter)+' -eps '+str(self.eps) +' -family '+str(familytpye)+' --pysal' + ' --constant'
            #generate faststwr-mpi.exe 
            mpi_cmd = 'mpiexec'+' -n '+ str(self.nproc) +' ' + os.getcwd() + '/fastmpiexc/dist/faststwr-mpi/' + 'faststwr-mpi.exe' + ' -DataFile ' +dfnames+' -Intervaldata '+dftime+' -out '+resultfile +' -schmethod '+self.search_method +' -criterion '+self.criterion+' -schtol '+str(self.tol)+' -max_iter '+str(self.max_iter)+' -eps '+str(self.eps) +' -family '+str(familytpye)+' --pysal' + ' --constant'
            if self.spherical:
                mpi_cmd += ' --spherical'           
            subprocess.run(mpi_cmd, shell=True)
            rsts = np.genfromtxt(resultfile, dtype=float, delimiter=',',skip_header=False)
            return rsts[0],rsts[1],int(rsts[2]),int(rsts[3])
           
class MPI_SelResults(object):
    def __init__(self, model, rslt):
        output = np.genfromtxt(rslt, dtype=float, delimiter=',',skip_header=False)
        self.k = model.X_list[-1].shape[1]+1
        self.n = model.coordslist[-1].shape[0]
        self.y = model.y_list[-1]
        self.index = output[:,0]
        self.influ = output[:,2]
        self.resid_response = output[:,1]
        self.params = output[:,3:(3+self.k)]
        self.CCT = output[:,-self.k:]
    
    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)
    
    @cache_readonly
    def bse(self):
        return np.sqrt(self.CCT*self.sigma2)
    
    @cache_readonly
    def sigma2(self):
        return (self.resid_ss / (self.n-self.tr_S))
    
    @cache_readonly
    def aicc(self):
        RSS = self.resid_ss
        trS = self.tr_S
        n = self.n
        aicc = n*np.log(RSS/n) + n*np.log(2*np.pi) + n*(n+trS)/(n-trS-2.0)
        return aicc
    
    @cache_readonly
    def resid_ss(self):
        return np.sum(self.resid_response**2)
    
    @cache_readonly
    def predy(self):
        return self.y.reshape(-1) - self.resid_response



################################################################################################    
#    def _init_sita_section(self,n_tick,gwr_bw0,brk_ticks):           
#            if(n_tick ==1):
#                sita_min = 0
#                sita_max = 0
#                sita_delta=0
#                mb_search = False
#            else: 
#                mb_search = True
#                tick_tims = self.tick_timesIntls[-n_tick:]
##                delt_t = np.sum(tick_tims)
#                lencd = self.coordslist[-1].shape[0]
#                lenwd = n_tick-1 
#                sita_arr  = np.zeros((lencd,lenwd))
#                gwr_bw0 =int(gwr_bw0) + 1
#                gwr_bwmin = self.coordslist[-1].shape[1]+1
#                dspmat0 =self.sorted_dspal_list[0][:,:gwr_bw0]
#                bw_0_lis =dspmat0[:,-1:].reshape((-1,1))
#                for i in range(n_tick-1):#最小值必须是最小行裁剪后矩阵最大值
#                    sorted_dspmat_min = self.sorted_dspal_list[-(i+1)][:,:gwr_bwmin]
#                    bw_tick_lis = sorted_dspmat_min[:,-1:].reshape((-1,1))
#                    delt_jundge = bw_0_lis <= bw_tick_lis
#                    if np.any(delt_jundge):
#                            sita_min = 0
#                            sita_max = 0
##                            sita_delta=0  
#                            mb_search = False
#                    else:
#                        tick_ticks = tick_tims[-(i+1):]
#                        delt_tic = np.sum(tick_ticks)
#                        delt_sita_tick =(bw_0_lis - bw_tick_lis)/delt_tic 
#                        sita_arr[:,i:i+1] = np.arctan(delt_sita_tick)
#                sita_max = sita_arr.min()
#                sita_min = 0
#                sita_delta = (sita_max - sita_min)/brk_ticks
#            return mb_search,sita_min, sita_max,sita_delta
################################################################################################    