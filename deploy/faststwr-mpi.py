#__author__ = "xiangq@uidaho.edu"
"""
   FastSTWR-MPI is an implemention of STWR based on Message Passing Interface(MPI).
   To initial and excute the code, we need install MPICH in windows or openmpi in Linux 
   and the mpi4py is needed.
  
   Parameters
   ---------- 
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
"""
import math
import numpy as np
from mpi4py import MPI
import argparse
from spglm.family import Gaussian, Binomial, Poisson
from spglm.iwls import iwls, _compute_betas_gwr
import sys
from spreg import user_output as USER
from scipy.spatial.distance import cdist,pdist
from math import radians, sin, cos, sqrt, asin,exp,atan,tan
import pickle
#import time

def read(fname):   
    alldatalist = []
    f = open(fname, 'r',encoding = 'utf-8') 
    for line in f:
        curfilename = line.rstrip('\n')
        alldatalist.append(np.genfromtxt(curfilename, dtype=float, delimiter=',',skip_header=False))     
    f.close()
    minbw0 = alldatalist[-1][:,3:].shape[1]+1
    maxbw0 = alldatalist[-1].shape[0]   
    return alldatalist,minbw0,maxbw0
def readInteval(fname):
    if pysal:
        input = np.genfromtxt(fname, dtype=float, delimiter=',',skip_header=False)
    else:
        input = np.genfromtxt(fname, dtype=float, delimiter=',',skip_header=True)
    return input
def check_data_valid(sdDatalist):
        rstype = 0
        y_arr = sdDatalist[-1][:,2].reshape((-1,1))
        X_arr = sdDatalist[-1][:,3:]
        try:              
            nt =  USER.check_arrays(y_arr, X_arr)
            USER.check_y(y_arr,nt)
        except:
            rstype = 2 
        tick_nums = len(sdDatalist)-1
        for p in range(tick_nums):
             X_arr = np.vstack((X_arr,sdDatalist[-(2+p)][:,3:]))
             y_arr = np.vstack((y_arr,sdDatalist[-(2+p)][:,2].reshape((-1,1))))
             try:              
                nt =  USER.check_arrays(y_arr, X_arr)
                USER.check_y(y_arr,nt)
             except:
                rstype = 2
        for x_iter in range(X_arr.shape[1]):
            jd_x_tick = X_arr[:,x_iter:(x_iter+1)].copy()
            jd_x_tick = (jd_x_tick != 0).astype(np.float32)
            jd_x_tick = np.sum(jd_x_tick)
            if(jd_x_tick< X_arr.shape[1]):
                 rstype = 2
        return rstype

def _init_goledn_section(n_tick,brk_ticks):
        if(n_tick ==1):
                mb_search = False
                minalpha = 0
                maxalpha = 0
                intervel = 0
        else: 
                mb_search = True
                minalpha = 0
                maxalpha = 1
                intervel = 1.0/brk_ticks
        return mb_search,minalpha,maxalpha,intervel
 
def golden_section(a, c, delta, function, tol=1.0e-6, max_iter=200, int_score=True):
        b = a + delta * np.abs(c-a)
        d = c - delta * np.abs(c-a)
        opt_val = None
        score = None
        diff = 1.0e9
        iters  = 0
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
            if rank == 0:
                if score_b <= score_d:
                    opt_score = score_b
                    opt_val = b
                    c = d
                    d = b
                    b = a + delta * np.abs(c-a)
                else:
                    opt_score = score_d
                    opt_val = d
                    a = b
                    b = d
                    d = c - delta * np.abs(c-a)
                diff = score_b - score_d
                score = opt_score
            b = comm.bcast(b,root=0)
            d = comm.bcast(d,root=0)
            opt_val = comm.bcast(opt_val,root=0)
            diff = comm.bcast(diff,root=0)
            score = comm.bcast(score,root=0)
        return opt_val, score 

def golden_section_cache(a, c, delta, function, tol=1.0e-6, max_iter=200, int_score=True):
        global cache_type
        if rank==0:
            if cache_type == 0:
                    cache_type =4
                    print("golden_section_cache change cache_type to ",cache_type)
            elif cache_type == 1:
                    cache_type =2
                    print("golden_section_cache change cache_type to ",cache_type)                   
        cache_type = comm.bcast(cache_type,root=0) 
        
        b = a + delta * np.abs(c-a)
        d = c - delta * np.abs(c-a)
        opt_val = None
        score = None
        diff = 1.0e9
        iters  = 0
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
            if rank == 0:
                if score_b <= score_d:
                    opt_score = score_b
                    opt_val = b
                    c = d
                    d = b
                    b = a + delta * np.abs(c-a)
                else:
                    opt_score = score_d
                    opt_val = d
                    a = b
                    b = d
                    d = c - delta * np.abs(c-a)
                diff = score_b - score_d
                score = opt_score
            b = comm.bcast(b,root=0)
            d = comm.bcast(d,root=0)
            opt_val = comm.bcast(opt_val,root=0)
            diff = comm.bcast(diff,root=0)
            score = comm.bcast(score,root=0)
        if rank==0:
            if cache_type == 3:
                    cache_type =1
                    print("golden_section_cache change cache_type to ",cache_type)
            else :
                    print("golden_section_cache Error cache_type  ",cache_type)                   
        cache_type = comm.bcast(cache_type,root=0)
        return opt_val, score   
    
def gg_stwr_section2(a, b, function,max_itr = 200, tol=1.0e-6): 

    invphi = (math.sqrt(5) - 1) / 2
    invphi2 = (3 - math.sqrt(5)) / 2 
    b-=1
    int_tol = 1
    diff_int = 10e6   
    opt_alpha = None
    opt_val = None
    score = None
    iters  = 0
    dict = {} 
    h = b-a
    max_iter = int(math.ceil(math.log(tol/h)/math.log(invphi))) 
    max_iter = max_iter if max_iter < max_itr else max_itr
    c = np.round(a + invphi2 * h)
    d = np.round(a + invphi * h)

    while (iters < max_iter)and (diff_int >= int_tol):       
        iters += 1 
        if c in dict:
            score_c = dict[c]
        else:
            if rank ==0:
                score_c,opt_alpha_c = function(c)                
                dict[c] = score_c
            else:
                dict[c] = function(c)            
        if d in dict:
            score_d = dict[d]
        else:
            if rank ==0:        
                score_d,opt_alpha_d = function(d)
                dict[d] = score_d
            else:
                 dict[d]=function(d)
        if rank == 0:
            if score_c is None and score_d is None :
                  continue
            if score_c <= score_d:
                if opt_val is not None:
                    diff_int = abs(opt_val - c)
                opt_val = c 
                opt_score = score_c
                opt_alpha = opt_alpha_c
                b = d
                d = c
                h = invphi*h
                c = np.round(a + invphi2 * h)
                score = opt_score
            else:
                if opt_val is not None:
                        diff_int = abs(opt_val - d)
                opt_val = d
                opt_score = score_d
                opt_alpha = opt_alpha_d  
                a = c
                c = d 
                h = invphi*h  
                d = np.round(a + invphi * h)               
                score = opt_score      
        c = comm.bcast(c,root=0)
        d = comm.bcast(d,root=0)             
        opt_val = comm.bcast(opt_val,root=0)
        score = comm.bcast(score,root=0)
        opt_alpha = comm.bcast(opt_alpha,root=0)
        diff_int= comm.bcast(diff_int,root=0)

    return np.round(opt_val, 2), score,opt_alpha#,output

def gg_stwr_section(a, c, delta, function, tol, max_iter):
    c-=1
    int_tol = 1
    diff_int = 10e6
    b = a #+ delta * np.abs(c-a) # min
    d = c #- delta * np.abs(c-a) # max
    opt_alpha = None
    opt_val = None
    score = None
    diff = 1.0e9
    iters  = 0
    dict = {} 
    while (np.abs(diff) > tol and iters < max_iter)and (diff_int >= int_tol):      
        iters += 1 
        if b in dict:
            score_b = dict[b]
        else:
            if rank ==0:    
                score_b,opt_alpha_b = function(b)
                dict[b] = score_b
            else:
                dict[b] = function(b)            
        if d in dict:
            score_d = dict[d]
        else:
            if rank ==0:        
                score_d,opt_alpha_d = function(d)
                dict[d] = score_d
            else:
                 dict[d]=function(d)
        if rank == 0:
            if score_b <= score_d:
                if opt_val is not None:
                    diff_int = abs(opt_val - b)
                opt_val = b 
                opt_score = score_b
                opt_alpha = opt_alpha_b
                c = d
                d = b
                b = np.round(a + delta * np.abs(c-a))
                diff = score_b - score_d           
                score = opt_score
            else:
                if opt_val is not None:
                        diff_int = abs(opt_val - d)
                opt_val = d
                opt_score = score_d
                opt_alpha = opt_alpha_d
                a = b
                b = d 
                d = np.round(c - delta * np.abs(c-a))
                diff = score_b - score_d           
                score = opt_score
            print("current b and d ,a,c:",b,d,a,c)        
        b = comm.bcast(b,root=0)
        d = comm.bcast(d,root=0)             
        opt_val = comm.bcast(opt_val,root=0)
        score = comm.bcast(score,root=0)
        opt_alpha = comm.bcast(opt_alpha,root=0)
        diff = comm.bcast(diff,root=0)
        diff_int= comm.bcast(diff_int,root=0)
    return np.round(opt_val, 2), score,opt_alpha

def equal_stwr_interval(l_bound, u_bound, interval, function, int_score=False):            
    opt_val  =  None
    opt_score = None
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a,0)
        c = np.round(c,0)
        b = np.round(b,0)
    score_a = function(a)
    score_c = function(c)
    
    if rank == 0:
        if score_a < score_c:
            opt_val = a
            opt_score = score_a
        else:
            opt_val = c
            opt_score = score_c
    while b < c:
        score_b = function(b)
        if rank == 0:
            if (score_b < opt_score): 
                opt_val = b
                opt_score = score_b
        b = comm.bcast(b,root=0)
        opt_val = comm.bcast(opt_val,root=0)
        opt_score = comm.bcast(opt_score,root=0)     
        b = b + interval
    return opt_val, opt_score

def equal_stwr_interval_cache(l_bound, u_bound, interval, function, int_score=False):            
    global cache_type
    if rank==0:
        if cache_type == 0:
                cache_type =4
                print("equal_stwr_interval_cache change cache_type to ",cache_type)
        elif cache_type == 1:
                cache_type =2
                print("equal_stwr_interval_cache change cache_type to ",cache_type)                   
    cache_type = comm.bcast(cache_type,root=0) 
    opt_val  =  None
    opt_score = None
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a,0)
        c = np.round(c,0)
        b = np.round(b,0)
    score_a = function(a)
    score_c = function(c)

    if rank == 0:
        if score_a < score_c:
            opt_val = a
            opt_score = score_a
        else:
            opt_val = c
            opt_score = score_c
    while b < c:
        score_b = function(b)
        if rank == 0:
            if (score_b < opt_score): 
                opt_val = b
                opt_score = score_b
        b = comm.bcast(b,root=0)
        opt_val = comm.bcast(opt_val,root=0)
        opt_score = comm.bcast(opt_score,root=0)     
        b = b + interval
    if rank==0:
            if cache_type == 3:
                    cache_type =1
                    print("equal_stwr_interval_cache change cache_type to ",cache_type)
            else :
                    print("equal_stwr_interval_cache Error cache_type  ",cache_type)                   
    cache_type = comm.bcast(cache_type,root=0)
    return opt_val, opt_score#, output


def _schgw0_func(tick_num,gwr_bw0_tick,_opt_bsita_tick = 0):  
        global cache_type
        if rank==0:
            if cache_type == 0:
                    cache_type =4
                    print("_schgw0_func change cache_type to ",cache_type)
            elif cache_type == 1:
                    cache_type =2
                    print("_schgw0_func change cache_type to ",cache_type)                   
        cache_type = comm.bcast(cache_type,root=0) 
   
        opt_score_tick = None
        opt_alpha_tick = None   
        gwr_bw0_tick = int(gwr_bw0_tick)     
        if rank ==0:
            opt_score_list_tick = []
            opt_alpha_list_tick =[]
        stwr_func = lambda alpha: mpi_CalCriterion(tick_num,_opt_bsita_tick,gwr_bw0_tick,alpha) 
        if  schmethod == 'golden_section':
                mb_search,alpha_min,alpha_max,alpha_interval=_init_goledn_section(tick_num,25)
                if mb_search:
                    alpha_interval = 0.38197 
                    try:
                        opt_alpha_tick, opt_score_tick = golden_section(alpha_min, alpha_max,alpha_interval,stwr_func, schtol,25,int_score = False)
                    except:
                        pass
                else: 
                    opt_alpha_tick =0
                    try:
                        opt_score_tick = stwr_func(0)
                    except:
                        pass
                if rank ==0: 
                    opt_score_list_tick.append(opt_score_tick)
                    opt_alpha_list_tick.append(opt_alpha_tick)
                   
        elif schmethod == 'interval':
                mb_search,alpha_min,alpha_max,alpha_interval= _init_alpha_search(tick_num,25)
                if (mb_search):
                    try :
                        opt_alpha_tick,opt_score_tick = equal_stwr_interval(alpha_min, alpha_max,alpha_interval,stwr_func)
                    except:
                        pass
                else: 
                    opt_alpha_tick =0
                    try:
                        opt_score_tick = stwr_func(0)
                    except:
                        pass
                if rank ==0:
                            opt_score_list_tick.append(opt_score_tick)
                            opt_alpha_list_tick.append(opt_alpha_tick)
        if rank==0:
            if cache_type == 3:
                    cache_type =1
                    print("_schgw0_func change cache_type to ",cache_type)
            else :
                    print("_schgw0_func Error cache_type  ",cache_type)                   
        cache_type = comm.bcast(cache_type,root=0)
        if rank ==0:        
                 index_tick = opt_score_list_tick.index(min(opt_score_list_tick))
                 opt_score_tick = opt_score_list_tick[index_tick]
                 opt_alpha_tick = opt_alpha_list_tick[index_tick]                
                 return opt_score_tick,opt_alpha_tick     
        return 
 
def _init_alpha_search(n_tick,brk_ticks): 
            if(n_tick ==1):
                mb_search = False
                minalpha = 0
                maxalpha = 0
                intervel = 0
            else: 
                mb_search = True
                minalpha = 0
                maxalpha = 1
                intervel = 1.0/brk_ticks
            return mb_search,minalpha,maxalpha,intervel    
def search_fit(minbw0,maxbw0,maxtimeIntervals):
     global cache_SDchunk 
     global cache_TDchunk 
     global cache_Mlist
     global cache_sortSD
     global cache_type
     print("maxtimeIntervals.shape",maxtimeIntervals.shape)
     tickused = maxtimeIntervals.shape[0]
     print("max_tickused",tickused)
     d_search_gw0 = maxbw0-minbw0+1
     if rank == 0:
            opt_score_list = []
            opt_bsita_list = []
            opt_tick_list = []
            opt_alpha_list =[]
            opt_gwr_bw0_list=[]
            print("Optimal Bandwidth Searching...")
     if(d_search_gw0 <= 100):
         for iticu in range(tickused):
             if rank==0:
                 cache_type=0     
             cache_type = comm.bcast(cache_type,root=0)
             cache_SDchunk =None 
             cache_TDchunk =None  
             cache_Mlist =None 
             cache_sortSD =None 
             print("tickused now is :",iticu+1)
             for q in range(d_search_gw0):                
                        gwr_bw0_tick = q+minbw0
                        _opt_bsita_tick = 0
                        stwr_func = lambda alpha: mpi_CalCriterion(iticu+1,_opt_bsita_tick,gwr_bw0_tick,alpha) 
                        if  schmethod == 'golden_section':
                                mb_search,alpha_min,alpha_max,alpha_interval=_init_goledn_section(iticu+1,25)
                                if mb_search:
                                    alpha_interval =  0.38197 
                                    try:
                                        opt_alpha_tick, opt_score = golden_section_cache(alpha_min, alpha_max,alpha_interval,stwr_func, schtol,max_iter,int_score = False)
                                    except:
                                        continue
                                else: 
                                    opt_alpha_tick =0
                                    try:
                                        opt_score = stwr_func(0)
                                    except:
                                        continue
                                if rank ==0:
                                    opt_bsita_list.append(_opt_bsita_tick)
                                    opt_score_list.append(opt_score)
                                    opt_tick_list.append(iticu+1)
                                    opt_alpha_list.append(opt_alpha_tick)
                                    opt_gwr_bw0_list.append(gwr_bw0_tick-1)#for stwr bandwidth will plus 1
                        elif schmethod == 'interval':
                                mb_search,alpha_min,alpha_max,alpha_interval= _init_alpha_search(iticu+1,25)
                                if (mb_search):
                                    try :
                                        opt_alpha_tick,opt_score = equal_stwr_interval_cache(alpha_min, alpha_max,alpha_interval,stwr_func)
                                    except:
                                        continue
                                else: 
                                    opt_alpha_tick =0
                                    try:
                                        opt_score = stwr_func(0)
                                    except:
                                        continue
                                if rank ==0: 
                                    opt_bsita_list.append(_opt_bsita_tick)
                                    opt_alpha_list.append(opt_alpha_tick)
                                    opt_score_list.append(opt_score)
                                    opt_tick_list.append(iticu+1)
                                    opt_gwr_bw0_list.append(gwr_bw0_tick-1)#for stwr bandwidth will plus 1
                                    
                        else:
                            raise TypeError('Unsupported computational search method ',schmethod)   
        
     else:
          if rank==0:
               cache_type=0  
               print("Search fit tpye change to ",cache_type)
          cache_type = comm.bcast(cache_type,root=0)
          cache_SDchunk =None 
          cache_TDchunk =None  
          cache_Mlist =None 
          cache_sortSD =None        
          _opt_bsita_tick = 0
          delta = 0.38197 
          opt_alpha_tick = 0
          stwr_func = lambda gwr_bw0_tick: mpi_CalCriterion(1,_opt_bsita_tick,gwr_bw0_tick,opt_alpha_tick) 
          gwr_bw0_tick, opt_score = golden_section(minbw0, maxbw0,delta,stwr_func,schtol,100)#200)     
          if rank ==0:
                        opt_bsita_list.append(_opt_bsita_tick)
                        opt_score_list.append(opt_score)
                        opt_tick_list.append(1)
                        opt_alpha_list.append(opt_alpha_tick)
                        opt_gwr_bw0_list.append(gwr_bw0_tick-1)          
          for iticu in range(tickused-1):
                     if rank==0:
                         cache_type=0  
                         print("search fit iticu tpye change to ",cache_type)
                     cache_type = comm.bcast(cache_type,root=0)                    
                     cache_SDchunk =None 
                     cache_TDchunk =None  
                     cache_Mlist =None 
                     cache_sortSD =None 
                     iticu += 1
                     print("tickused now is :",iticu+1)
                     optbwfunc = lambda gwr_bw0:_schgw0_func(iticu+1,gwr_bw0,_opt_bsita_tick)
                     opt_bw0_tick, opt_score,opt_alpha = gg_stwr_section2(minbw0, maxbw0,optbwfunc)   
                     if rank ==0:
                             opt_bsita_list.append(_opt_bsita_tick)
                             opt_alpha_list.append(opt_alpha)
                             opt_score_list.append(opt_score)
                             opt_tick_list.append(iticu+1)
                             opt_gwr_bw0_list.append(opt_bw0_tick-1)                          
     if rank ==0:
         print("Searching ended ")
         index = opt_score_list.index(min(opt_score_list))          
         savedata = np.array((opt_alpha_list[index],opt_bsita_list[index],opt_tick_list[index],opt_gwr_bw0_list[index]))
         print("opt_alpha is:",opt_alpha_list[index])
         print("opt_sita is :",opt_bsita_list[index])
         print("opt_tick is :",opt_tick_list[index])
         print("opt_bw0:",opt_gwr_bw0_list[index])        
         print ("opt_score and index,optalpha ",index,opt_score_list[index],opt_alpha_list[index])    
         np.savetxt(fout, savedata,delimiter=',',comments='')
    
def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c              
        
def local_CalSearch(X_curtol,y_curtol,i,opt_bsita,gwr_bw0,alpha,subData,subIvls,tick_time_used,final = False,sph=False):   
    lasttolCoord = Coordlist[-1]  
    lastcoord_cal = lasttolCoord[i] 
    lasty_cal = y_curtol[i]
    gwr_bw0 = int(gwr_bw0)
    spdist_rt =None
    tmdist_rt =None   
    if sph: 
        sdistance = np.zeros_like(lasttolCoord[:,1])
        idxd = 0
        for lpb in lasttolCoord:
                sdistance[idxd] = _haversine(lastcoord_cal[0],lastcoord_cal[1],lpb[0],lpb[1])
                idxd += 1
        spdist_rt = sdistance.copy()
        spdist_rt = spdist_rt.reshape((1,-1))
        tdistance =  np.zeros_like(sdistance)
        tmdist_rt = tdistance.copy()
        tmdist_rt = tmdist_rt.reshape((1,-1))
        delta_t_total =np.sum(subIvls)*1.0 
        #trunck at t0 
        spt_check = np.unique(np.partition(sdistance, gwr_bw0)[:gwr_bw0])#*eps
        sbwsize = spt_check.shape[0]
        expandbw = gwr_bw0
        while sbwsize<mcal_minbw:
            expandbw += 1
            spt_check_tick= np.unique(np.partition(sdistance,expandbw)[:expandbw])
            sbwsize = spt_check_tick.shape[0]
        maxsd0 = np.partition(sdistance,expandbw)[expandbw]*eps      
        sdistance = sdistance/maxsd0
        sdistance[sdistance>=1] = 1
        sdistance = sdistance.reshape((-1,1))
        tdistance = tdistance.reshape((-1,1))
        m_randomeps =  np.absolute(np.random.normal(0, m_eps,sdistance.shape))
        sdistance += m_randomeps
        for p in range(tick_time_used-1):
                delt_tick_tol = np.sum(subIvls[:(p+2)])
                tick_coords = Coordlist[-(p+2)]
                tick_y = subData[-(p+2)][:,2]
                stic_distance = np.zeros_like(tick_y)
                ttic_distance = np.zeros_like(tick_y) 
                idxd = 0
                for lpb in tick_coords:
                        if tick_y[idxd] ==0:
                            tick_y[idxd] =0.000001              
                        stic_distance[idxd] = _haversine(lastcoord_cal[0],lastcoord_cal[1],lpb[0],lpb[1])
                        ttic_distance[idxd] =  delta_t_total*abs((tick_y[idxd]-lasty_cal)/tick_y[idxd])/delt_tick_tol
                        idxd += 1
                stic_distance = np.asarray(stic_distance).reshape((-1))
                ttic_distance = np.asarray(ttic_distance).reshape((-1))                
                spdist_rt_tick = stic_distance.copy()
                spdist_rt_tick = spdist_rt_tick.reshape((1,-1))
                spdist_rt = np.hstack((spdist_rt,spdist_rt_tick))
                tmdist_rt_tick = ttic_distance.copy()
                tmdist_rt_tick = tmdist_rt_tick.reshape((1,-1))
                tmdist_rt = np.hstack((tmdist_rt,tmdist_rt_tick))               
                spt_check = np.unique(np.partition(stic_distance, mcal_minbw)[:mcal_minbw])#*eps
                sbwsize = spt_check.shape[0]
                expandbw = mcal_minbw
                while sbwsize<mcal_minbw:
                    expandbw += 1
                    spt_check_tick= np.unique(np.partition(stic_distance,expandbw)[:expandbw])
                    sbwsize = spt_check_tick.shape[0]
                maxsd_tick = np.partition(stic_distance,expandbw)[expandbw]*eps
                gwr_bw_tick = maxsd0 - tan(opt_bsita)*delt_tick_tol#subIvls[(p+1)]
                if maxsd_tick < gwr_bw_tick:
                        maxsd_tick = gwr_bw_tick            
                stic_distance = stic_distance/maxsd_tick
                stic_distance = stic_distance.reshape((-1,1))
                ttic_distance = ttic_distance.reshape((-1,1))             
                ttic_distance[stic_distance>=1] = 0
                stic_distance[stic_distance>=1] = 1                
                ttic_distance += np.absolute(np.random.normal(0, m_eps,ttic_distance.shape))
                stic_distance += np.absolute(np.random.normal(0, m_eps,ttic_distance.shape))         
                sdistance = np.vstack((sdistance,stic_distance))
                tdistance = np.vstack((tdistance,ttic_distance))
    else:         
        sdistance = cdist([lastcoord_cal],lasttolCoord)      
        sdistance = np.asarray(sdistance).reshape((-1)) 
        spdist_rt = sdistance.copy()
        spdist_rt = spdist_rt.reshape((1,-1))
        
        spt_check = np.unique( np.partition(sdistance,gwr_bw0)[:gwr_bw0] )#*eps
        expandbw = gwr_bw0
        sbwsize = spt_check.shape[0]      
        while sbwsize<mcal_minbw:
            expandbw += 1
            spt_check_tick= np.unique(np.partition(sdistance,expandbw)[:expandbw])
            sbwsize = spt_check_tick.shape[0]
        maxsd0 = np.partition(sdistance,expandbw)[expandbw]*eps  
     
        sdistance = sdistance/maxsd0
        sdistance[sdistance>=1] = 1
        sdistance += np.absolute(np.random.normal(0, m_eps,sdistance.shape))
        sdistance = sdistance.reshape((-1,1))
        tdistance = np.zeros_like(sdistance)
        tmdist_rt = tdistance.copy()
        tmdist_rt = tmdist_rt.reshape((1,-1))
        delta_t_total =np.sum(subIvls)*1.0
           
        for p in range(tick_time_used-1):
                delt_tick_tol = np.sum(subIvls[:(p+2)])
                tick_coords = Coordlist[-(p+2)]
                tick_y = subData[-(p+2)][:,2]
                stic_distance = cdist([lastcoord_cal],tick_coords)
                stic_distance = np.asarray(stic_distance).reshape((-1))
                ttic_distance = np.zeros_like(stic_distance)    
                idxd = 0
                for lpb in tick_coords:
                        ttic_distance[idxd] =  delta_t_total*abs((tick_y[idxd]-lasty_cal)/tick_y[idxd])/delt_tick_tol
                        idxd += 1
                spdist_rt_tick = stic_distance.copy()
                spdist_rt_tick = spdist_rt_tick.reshape((1,-1))
                spdist_rt = np.hstack((spdist_rt,spdist_rt_tick))
                tmdist_rt_tick = ttic_distance.copy()
                tmdist_rt_tick = tmdist_rt_tick.reshape((1,-1))
                tmdist_rt = np.hstack((tmdist_rt,tmdist_rt_tick))               
                spt_check = np.unique(np.partition(stic_distance, mcal_minbw)[:mcal_minbw])
                sbwsize = spt_check.shape[0]
                expandbw = mcal_minbw
                while sbwsize<mcal_minbw:
                    expandbw += 1
                    spt_check_tick= np.unique(np.partition(stic_distance,expandbw)[:expandbw])
                    sbwsize = spt_check_tick.shape[0]
                maxsd_tick = np.partition(stic_distance,expandbw)[expandbw]*eps
                gwr_bw_tick = maxsd0 - tan(opt_bsita)*delt_tick_tol
                if maxsd_tick < gwr_bw_tick:
                        maxsd_tick = gwr_bw_tick
                stic_distance = stic_distance/maxsd_tick
                stic_distance = stic_distance.reshape((-1,1))
                ttic_distance = ttic_distance.reshape((-1,1))
                ttic_distance[stic_distance>=1] = 0
                stic_distance[stic_distance>=1] = 1
                ttic_distance += np.absolute(np.random.normal(0, m_eps,ttic_distance.shape))
                stic_distance += np.absolute(np.random.normal(0, m_eps,ttic_distance.shape))
                sdistance = np.vstack((sdistance,stic_distance))

                tdistance = np.vstack((tdistance,ttic_distance))
    
    m_dtm0 = False
    if(tick_time_used == 1):
        m_dtm0 = True 
    wi  = np.zeros_like(sdistance)
    len_last = len(lasttolCoord) 
    wi[:len_last,:] = cal_stWeight(sdistance[:len_last,:],tdistance[:len_last,:],alpha,m_dtm0 = True)
    wi[len_last:,:] = cal_stWeight(sdistance[len_last:,:],tdistance[len_last:,:],alpha,m_dtm0)
    if final:
        if familytpye==0:
            cur_family = Gaussian()
        else:
            cur_family = Poisson()
        cur_offset = np.ones((y_curtol.shape[0],1))
        rslt = iwls(y_curtol,X_curtol,cur_family,cur_offset,None,None,1.0e-5,20,wi)
        params = rslt[0].T
        predy = rslt[1][i]
        w = rslt[3][i]
        S = np.dot(X_curtol,rslt[5])
        CCT = np.diag(np.dot(rslt[5],rslt[5].T))
        return np.concatenate((i,w,predy,params,CCT,S,wi.reshape((1,-1))))
    else:    

        if criterion == 'STWR_CV':#calculate the CV scores    
             betas, inv_xtx_xt = _compute_betas_gwr(y_curtol, X_curtol, wi)
             influ = np.dot(X_curtol[i], inv_xtx_xt[:, i])
             predy = np.dot(X_curtol[i], betas)[0]
             resid = y_curtol[i] - predy
             aa = resid/(1.0-influ)
             return aa**2,spdist_rt,tmdist_rt
        elif criterion == 'Stwr_AICc':#calculate the AICc scores  
                X_new = X_curtol*np.sqrt(wi) 
                Y_new = y_curtol*np.sqrt(wi) 
                temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
                hat =  np.dot(X_new[i],temp[:,i])
                yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
                err = Y_new[i][0]-yhat
                return err*err,hat,spdist_rt,tmdist_rt 
        else:
            raise TypeError('Unsupported criterion :',criterion)  
  
def cal_stWeight(sdistance,tdistance,alpha,m_dtm0 = False,kername ='spt_bisquare'): 
     if m_dtm0:   
            if kername =='spt_bisquare' :
                         return (1-(sdistance)**2)**2
     else:          
             if kername =='spt_bisquare' :
                     #return ((1-alpha)*((1-(sdistance)**2)**2)+alpha*(2/(1+np.exp(-tdistance))-1))/2   
                     return ((1-alpha)*((1-(sdistance)**2)**2)+alpha*(2/(1+np.exp(-tdistance))-1))          
     
def mpi_CalCriterion(tick_time_used,_opt_bsita,gwr_bw0,alpha,final= False):          
    global cache_SDchunk 
    global cache_TDchunk 
    global cache_Mlist
    global cache_sortSD
    global cache_type
    global cache_MSDchunk
    global cache_MTDchunk       
    gwr_bw0 = int(gwr_bw0)
    X_curtol = X_tollist[-1]
    y_curtol = y_tollist[-1]
    n_curtol = n_tollist[-1]
    for sq in range(tick_time_used-1):
            X_curtol = np.vstack((X_curtol,X_tollist[-(sq+2)]))
            y_curtol = np.vstack((y_curtol,y_tollist[-(sq+2)]))          
    m_autoChange = 1
    if cache_type == 4:
                m_autoChange = 2
                cache_type = 0
    if cache_SDchunk is None or cache_Mlist is None or cache_sortSD is None :
                    cache_type =0
    if cache_type ==0:#No any content cached and its unstable. Both SD and TD need to be calculated and cached then cache_type will be set to 1.
            subtct_Data = sdDatalist[-tick_time_used:]
            subtct_Intervals= sdTIntval[:tick_time_used]
            cache_Mlist = [n_curtol]
            for sq in range(tick_time_used-1):
                cache_Mlist.append(n_tollist[-(sq+2)])      
            if final:
                sub_Betas = np.empty((chunk.shape[0],2*X_curtol.shape[1]+3+y_curtol.shape[0]*2), dtype=np.float32)
                pos = 0
                for i in chunk:
                    sub_Betas[pos] = local_CalSearch(X_curtol,y_curtol,i,_opt_bsita,gwr_bw0,alpha,subtct_Data,subtct_Intervals,tick_time_used,final,sph = spherical)#local_CalSearch(i,bw,fixed,final)
                    pos+=1
                Betas_list = comm.gather(sub_Betas, root=0)
                
                if rank ==0:
                    data = np.vstack(Betas_list)
                    if pysal:
                        np.savetxt(fout, data, delimiter=',',comments='')
                    if not pysal:
                        header="i,w,predy,params,CCT,S,wi,"
                        varNames = np.genfromtxt(fDatafname, dtype=str, delimiter=',',names=True,max_rows=1).dtype.names[3:]
                        if constant:
                            varNames = ['intercept'] + list(varNames)
                        for x in varNames:
                            header += ("b_"+x+',')
                        for x in varNames:
                            header += ("se_"+x+',')
                            np.savetxt(fout, data, delimiter=',',header=header[:-1],comments='')
                    return
                return

            if criterion == 'STWR_CV':
                    sub_cv = 0
                    m_sizetol = np.sum(cache_Mlist)
                    n_sizelen = chunk.shape[0]         
                    cache_SDchunk = np.zeros((n_sizelen,m_sizetol))
                    cache_TDchunk = np.zeros((n_sizelen,m_sizetol))
                    
                    cache_sortSD  = []
                    ti_idx = 0
                    for i in chunk:
                         aa2,spdtol,tmdtol = local_CalSearch(X_curtol,y_curtol,i,_opt_bsita,gwr_bw0,alpha,subtct_Data,subtct_Intervals,tick_time_used,final,sph = spherical)                 
                         sub_cv += aa2
                         cache_SDchunk[ti_idx] = spdtol
                         cache_TDchunk[ti_idx] = tmdtol
                         ti_idx += 1
                    ti_idx=0
                    for ti in cache_Mlist:
                        cache_sortSD.append(np.sort(cache_SDchunk[:,ti_idx:ti_idx+ti]))
                        ti_idx += ti  
                    cache_type = m_autoChange
                    CV_list = comm.gather(sub_cv, root=0)          
                    if rank==0: 
                         CV_score = sum(CV_list)/n_curtol
                         print ("tick_time_used,gwr_bw0,alpha, CV_score are:",tick_time_used,gwr_bw0,alpha, CV_score)    
                         return CV_score
                    return
            elif criterion == 'Stwr_AICc':
                    sub_RSS = 0
                    sub_trS = 0 
                    m_sizetol = np.sum(cache_Mlist)
                    n_sizelen = chunk.shape[0]         
                    cache_SDchunk = np.zeros((n_sizelen,m_sizetol))
                    cache_TDchunk = np.zeros((n_sizelen,m_sizetol))         
      
                    cache_sortSD  = []
                    ti_idx = 0
                    for i in chunk:
                          err2,hat,spdtol,tmdtol = local_CalSearch(X_curtol,y_curtol,i,_opt_bsita,gwr_bw0,alpha,subtct_Data,subtct_Intervals,tick_time_used,final,sph = spherical)                 
                          sub_RSS += err2
                          sub_trS += hat 
                          cache_SDchunk[ti_idx] = spdtol
                          cache_TDchunk[ti_idx] = tmdtol
                          ti_idx += 1
                    ti_idx=0
                    for ti in cache_Mlist:
                        cache_sortSD.append(np.sort(cache_SDchunk[:,ti_idx:ti_idx+ti]))
                        ti_idx += ti  
                    cache_type = m_autoChange
         
                    RSS_list = comm.gather(sub_RSS, root=0)
                    trS_list = comm.gather(sub_trS, root=0) 
                    if rank==0:
                        tot_RSS = sum(RSS_list)
                        tot_trS = sum(trS_list)
                        print("y_curtol shape",y_curtol.shape)
                        y_cal_tol = y_curtol[:n_curtol]
                        print("y_cal_tol shape",y_cal_tol.shape)
                        tot_TSS = np.sum((y_cal_tol - np.mean(y_cal_tol))**2)
                        llf = -np.log(tot_RSS)*n_curtol/2 - (1+np.log(np.pi/n_curtol*2))*n_curtol/2
                        aicc = -2*llf + 2.0*n_curtol*(tot_trS + 1.0)/(n_curtol-tot_trS-2.0)                   
                        R2 = 1- tot_RSS/tot_TSS
                        print("tick_time_used,BW, llf,R2,alpha,tot_trS are:",tick_time_used,gwr_bw0,aicc,R2,alpha,tot_trS)     
                        return aicc
                    return
            else:
                     raise TypeError('Unsupported criterion :',criterion)                       
    elif cache_type ==1:#Distance matrix is cached but we need adjust the bandwidth to re-calculate trunced distance matrix （Default status）  
            subtct_Intervals= sdTIntval[:tick_time_used]
            n_curtol = n_tollist[-1]
            cal_W = np.zeros(cache_SDchunk.shape)
            dspmat0 =cache_sortSD[0][:,:gwr_bw0]
            dspmat_tick = dspmat0[:,1:2] 
            dspmat_last = dspmat0[:,-1:]
            delt_jundge = dspmat_tick == dspmat_last
            add_neighbor=1
            while np.any(delt_jundge):
                dspmat0 =cache_sortSD[0][:,:gwr_bw0+add_neighbor]
                dspmat_tick = dspmat0[:,1:2] 
                dspmat_last = dspmat0[:,-1:]
                delt_jundge = dspmat_tick == dspmat_last
                add_neighbor = add_neighbor+1 
            mask_tick = dspmat0.max(axis=1) * eps
            mask_tick.shape = (dspmat0.shape[0],1)
            distgwrb0 = mask_tick.copy() 
            mask_tick = np.repeat(mask_tick,cache_Mlist[0], axis=1)
            cal_spl = cache_SDchunk[:,:cache_Mlist[0]]/mask_tick
            cal_spl[cal_spl>=1]=1
            cal_spl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))
            cal_tpl = cache_TDchunk[:,:cache_Mlist[0]].copy()
            cal_tpl[cal_spl>=1]=0
            cal_tpl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))   
            m_dtm0 = False
            if(tick_time_used == 1):
                    m_dtm0 = True  
            cal_W[:,:cache_Mlist[0]] = cal_stWeight(cal_spl,cal_tpl,alpha,m_dtm0=True) 
            tuid = cache_Mlist[0]
            for tu in range(tick_time_used-1):
                 distgwr_tick=  distgwrb0 - tan(_opt_bsita)* subtct_Intervals[-(i+1)]
                 sorted_dspmat_min = cache_sortSD[tu+1][:,:mcal_minbw]
                 dspmat_tick = sorted_dspmat_min[:,1:2]
                 dspmat_last = sorted_dspmat_min[:,-1:]
                 delt_jundge_same = dspmat_tick == dspmat_last
                 add_neighbor=1
                 while np.any(delt_jundge_same):
                      sorted_dspmat_min = cache_sortSD[tu+1][:,:mcal_minbw+add_neighbor]
                      dspmat_tick = sorted_dspmat_min[:,1:2] 
                      dspmat_last = sorted_dspmat_min[:,-1:]
                      delt_jundge_same = dspmat_tick == dspmat_last
                      add_neighbor = add_neighbor+1 
                 cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                 delt_jundge = distgwr_tick < cal_bw_min
                 if np.any(delt_jundge):
                     np.copyto(distgwr_tick,cal_bw_min* eps,where=delt_jundge)
                 mask_tick = np.repeat(distgwr_tick,cache_Mlist[tu+1], axis=1)                
                 cal_spl = cache_SDchunk[:,tuid:cache_Mlist[tu+1]]/mask_tick
                 cal_spl[cal_spl>=1]=1
                 cal_tpl = cache_TDchunk[:,tuid:cache_Mlist[tu+1]].copy()
                 cal_tpl[cal_spl>=1]=0
                 cal_spl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))
                 cal_tpl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))
                 cal_W[:,tuid:cache_Mlist[tu+1]] = cal_stWeight(cal_spl,cal_tpl,alpha)  
                 tuid += cache_Mlist[tu+1]
            if criterion == 'STWR_CV':
                sub_cv = 0
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    betas, inv_xtx_xt = _compute_betas_gwr(y_curtol, X_curtol, wi)
                    influ = np.dot(X_curtol[i], inv_xtx_xt[:, i])
                    resid = y_curtol[i] - np.dot(X_curtol[i], betas)[0]
                    aa = resid/(1.0-influ)
                    sub_cv += aa**2
                    ci += 1
                CV_list = comm.gather(sub_cv, root=0)          
                if rank==0:     
                     CV_score = sum(CV_list)/n_curtol
                     print ("cache_type,tick_time_used,gwr_bw0,alpha, CV_score are:",cache_type,tick_time_used,gwr_bw0,alpha, CV_score)  
                     return CV_score     
                return
            elif criterion == 'Stwr_AICc':
                sub_RSS = 0
                sub_trS = 0 
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    X_new = X_curtol*np.sqrt(wi) 
                    Y_new = y_curtol*np.sqrt(wi) 
                    temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
                    hat =  np.dot(X_new[i],temp[:,i])
                    yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
                    err = Y_new[i][0]-yhat
                    sub_RSS += err*err
                    sub_trS += hat
                    ci += 1
                
                RSS_list = comm.gather(sub_RSS, root=0)
                trS_list = comm.gather(sub_trS, root=0) 
                if rank==0:
                    tot_RSS = sum(RSS_list)
                    tot_trS = sum(trS_list)
                    print("y_curtol shape",y_curtol.shape)
                    y_cal_tol = y_curtol[:n_curtol]
                    print("y_cal_tol shape",y_cal_tol.shape)
                    tot_TSS = np.sum((y_cal_tol - np.mean(y_cal_tol))**2)
                    llf = -np.log(tot_RSS)*n_curtol/2 - (1+np.log(np.pi/n_curtol*2))*n_curtol/2
                    aicc = -2*llf + 2.0*n_curtol*(tot_trS + 1.0)/(n_curtol-tot_trS-2.0)
                    R2 = 1- tot_RSS/tot_TSS
                    print("cache_type,tick_time_used,gwr_bw0,alpha,Aicc are:",cache_type,tick_time_used,gwr_bw0,alpha, aicc)     
                    return aicc
                return
            else:
                     raise TypeError('Unsupported criterion :',criterion)               
    elif cache_type ==2:#Recalculate distance after mask and set the cache_type to 3.
            print ("cache_type ",cache_type)
            cache_MSDchunk  =  cache_SDchunk.copy()
            cache_MTDchunk  =  cache_TDchunk.copy()            
            subtct_Intervals= sdTIntval[:tick_time_used]
            cal_W = np.zeros(cache_SDchunk.shape)
            dspmat0 =cache_sortSD[0][:,:gwr_bw0]
            dspmat_tick = dspmat0[:,1:2] 
            dspmat_last = dspmat0[:,-1:]
            delt_jundge = dspmat_tick == dspmat_last
            add_neighbor=1         
            while np.any(delt_jundge):
                dspmat0 =cache_sortSD[0][:,:gwr_bw0+add_neighbor]
                dspmat_tick = dspmat0[:,1:2] 
                dspmat_last = dspmat0[:,-1:]
                delt_jundge = dspmat_tick == dspmat_last
                add_neighbor = add_neighbor+1           
            mask_tick = dspmat0.max(axis=1) * eps
            mask_tick.shape = (dspmat0.shape[0],1)
            distgwrb0 = mask_tick.copy()           
            mask_tick = np.repeat(mask_tick,cache_Mlist[0], axis=1)     
            cal_spl = cache_MSDchunk[:,:cache_Mlist[0]]/mask_tick
            cal_spl[cal_spl>=1]=1     
            cal_tpl = cache_MTDchunk[:,:cache_Mlist[0]]
            cal_tpl[cal_spl>=1]=0
            cal_spl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))
            #update the cached chunk cache_MSDchunk
            cache_MSDchunk[:,:cache_Mlist[0]] = cal_spl
            cal_tpl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))

            m_dtm0 = False
            if(tick_time_used == 1):
                    m_dtm0 = True       
            cal_W[:,:cache_Mlist[0]] = cal_stWeight(cal_spl,cal_tpl,alpha,m_dtm0=True)
            tuid = cache_Mlist[0]
            for tu in range(tick_time_used-1):
                 distgwr_tick=  distgwrb0 - tan(_opt_bsita)* subtct_Intervals[-(tu+1)]
                 sorted_dspmat_min = cache_sortSD[tu+1][:,:mcal_minbw]
                 dspmat_tick = sorted_dspmat_min[:,1:2]
                 dspmat_last = sorted_dspmat_min[:,-1:]
                 delt_jundge_same = dspmat_tick == dspmat_last
                 add_neighbor=1 
                 while np.any(delt_jundge_same):
                      sorted_dspmat_min = cache_sortSD[tu+1][:,:mcal_minbw+add_neighbor]
                      dspmat_tick = sorted_dspmat_min[:,1:2] 
                      dspmat_last = sorted_dspmat_min[:,-1:]
                      delt_jundge_same = dspmat_tick == dspmat_last
                      add_neighbor = add_neighbor+1 
                 cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                 delt_jundge = distgwr_tick < cal_bw_min

                 if np.any(delt_jundge):
                     np.copyto(distgwr_tick,cal_bw_min* eps,where=delt_jundge)                              
                 mask_tick = np.repeat(distgwr_tick,cache_Mlist[tu+1], axis=1)
                 cal_spl = cache_MSDchunk[:,tuid:tuid+cache_Mlist[tu+1]]/mask_tick
                 cal_spl[cal_spl>=1]=1
                 cal_tpl = cache_MTDchunk[:,tuid:tuid+cache_Mlist[tu+1]]
                 cal_tpl[cal_spl>=1]=0
                 cal_spl += np.absolute(np.random.normal(0,m_eps,cal_spl.shape))
                 cal_tpl += np.absolute(np.random.normal(0,m_eps,cal_tpl.shape))
                 #update the cached chunk cache_SDchunk
                 cache_MSDchunk[:,tuid:tuid+cache_Mlist[tu+1]] = cal_spl
                 cal_W[:,tuid:tuid+cache_Mlist[tu+1]] = cal_stWeight(cal_spl,cal_tpl,alpha)  
                 tuid += cache_Mlist[tu+1]

            if criterion == 'STWR_CV':              
                sub_cv = 0
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    betas, inv_xtx_xt = _compute_betas_gwr(y_curtol, X_curtol, wi)
                    influ = np.dot(X_curtol[i], inv_xtx_xt[:, i])
                    resid = y_curtol[i] - np.dot(X_curtol[i], betas)[0]
                    aa = resid/(1.0-influ)
                    sub_cv += aa**2
                    ci += 1
                cache_type = 3 
                CV_list = comm.gather(sub_cv, root=0)   
                if rank==0:     
                     CV_score = sum(CV_list)/n_curtol
                     print ("2222,tick_time_used,gwr_bw0,alpha, CV_score are:",tick_time_used,gwr_bw0,alpha, CV_score)  
                     return CV_score
                return
            elif criterion == 'Stwr_AICc':
                sub_RSS = 0
                sub_trS = 0 
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    X_new = X_curtol*np.sqrt(wi) 
                    Y_new = y_curtol*np.sqrt(wi) 
                    temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
                    hat =  np.dot(X_new[i],temp[:,i])
                    yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
                    err = Y_new[i][0]-yhat
                    sub_RSS += err*err
                    sub_trS += hat
                    ci += 1     
                cache_type = 3  
                RSS_list = comm.gather(sub_RSS, root=0)
                trS_list = comm.gather(sub_trS, root=0) 
                if rank==0:
                    tot_RSS = sum(RSS_list)
                    tot_trS = sum(trS_list)
                    print("y_curtol shape",y_curtol.shape)
                    y_cal_tol = y_curtol[:n_curtol]
                    print("y_cal_tol shape",y_cal_tol.shape)
                    tot_TSS = np.sum((y_cal_tol - np.mean(y_cal_tol))**2)
                    llf = -np.log(tot_RSS)*n_curtol/2 - (1+np.log(np.pi/n_curtol*2))*n_curtol/2
                    aicc = -2*llf + 2.0*n_curtol*(tot_trS + 1.0)/(n_curtol-tot_trS-2.0)                  
                    R2 = 1- tot_RSS/tot_TSS
                    print("2222,cache_type,tick_time_used,gwr_bw0,alpha,Aicc are:",cache_type,tick_time_used,gwr_bw0,alpha, aicc)     
                    return aicc

                return 
            else:
                     raise TypeError('Unsupported criterion :',criterion)                 
    elif cache_type ==3:#缓存了distance,只需要调整alpha重新计算weight和score 
            cal_W = cal_stWeight(cache_MSDchunk,cache_MTDchunk,alpha)
            cal_W +=  np.absolute(np.random.normal(0,m_eps,cal_W.shape))
            if criterion == 'STWR_CV':       
                sub_cv = 0
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    betas, inv_xtx_xt = _compute_betas_gwr(y_curtol, X_curtol, wi)
                    influ = np.dot(X_curtol[i], inv_xtx_xt[:, i])
                    resid = y_curtol[i] - np.dot(X_curtol[i], betas)[0]
                    aa = resid/(1.0-influ)
                    sub_cv += aa**2
                    ci += 1
                CV_list = comm.gather(sub_cv, root=0)   
                if rank==0:     
                     CV_score = sum(CV_list)/n_curtol
                     print ("cache_type,tick_time_used,gwr_bw0,alpha, CV_score are:",cache_type,tick_time_used,gwr_bw0,alpha, CV_score) 
                     return CV_score
                return
            elif criterion == 'Stwr_AICc':
                sub_RSS = 0
                sub_trS = 0 
                ci = 0
                for i in chunk:
                    wi = cal_W[ci].reshape((-1, 1))
                    X_new = X_curtol*np.sqrt(wi) 
                    Y_new = y_curtol*np.sqrt(wi) 
                    temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
                    hat =  np.dot(X_new[i],temp[:,i])
                    yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
                    err = Y_new[i][0]-yhat
                    sub_RSS += err*err
                    sub_trS += hat
                    ci += 1     
                RSS_list = comm.gather(sub_RSS, root=0)
                trS_list = comm.gather(sub_trS, root=0) 
                if rank==0:
                    tot_RSS = sum(RSS_list)
                    tot_trS = sum(trS_list)
                    print("y_curtol shape",y_curtol.shape)
                    y_cal_tol = y_curtol[:n_curtol]
                    print("y_cal_tol shape",y_cal_tol.shape)
                    tot_TSS = np.sum((y_cal_tol - np.mean(y_cal_tol))**2)
                    llf = -np.log(tot_RSS)*n_curtol/2 - (1+np.log(np.pi/n_curtol*2))*n_curtol/2
                    aicc = -2*llf + 2.0*n_curtol*(tot_trS + 1.0)/(n_curtol-tot_trS-2.0)
                    R2 = 1- tot_RSS/tot_TSS
                    print("cache_type,tick_time_used,gwr_bw0,alpha,Aicc are:",cache_type,tick_time_used,gwr_bw0,alpha, aicc)     
                    return aicc
                return
            else:
                     raise TypeError('Unsupported criterion :',criterion)   
            
    else:
            print("Error cache_type",cache_type)
            return
    
if __name__ == "__main__":
    #Initializing
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print("%d of %d" % (comm.Get_rank(), size))   
    parser = argparse.ArgumentParser()
    parser.add_argument("-DataFile")
    parser.add_argument("-Intervaldata")
    parser.add_argument("-out")
    parser.add_argument("-schmethod")
    parser.add_argument("-criterion")
    parser.add_argument("-schtol")
    parser.add_argument("-eps")
    parser.add_argument("-max_iter")
    parser.add_argument("-family")
    parser.add_argument('-p','--pysal',action='store_true')
    parser.add_argument('-c','--constant',action='store_true')
    parser.add_argument('-s','--spherical',action='store_true')
    fDatafname  = parser.parse_args().DataFile
    fIntervalfname = parser.parse_args().Intervaldata
    fout  = parser.parse_args().out
    pysal = parser.parse_args().pysal
    constant = parser.parse_args().constant 
    spherical = parser.parse_args().spherical 
    familytpye = None
    
    if  parser.parse_args().family is None:
        familytpye = 0
    else:
        familytpye = int(parser.parse_args().family)
    if parser.parse_args().schmethod is None:  
        schmethod = 'interval'
    else:
        schmethod = parser.parse_args().schmethod   
    if parser.parse_args().criterion is None:   
        criterion='STWR_CV'
    else:
        criterion = parser.parse_args().criterion
    if  parser.parse_args().schtol is None:
        schtol = 1.0e-6 
    else:
        schtol = float(parser.parse_args().schtol)
    if  parser.parse_args().eps is None:
        eps = 1.0000001
    else:
        eps = float(parser.parse_args().eps)
    m_eps = eps -1
    max_iter = 200
    if parser.parse_args().max_iter is not None:
        max_iter = int(parser.parse_args().max_iter)
    if spherical:  
        spherical = True
    else:
        spherical = False 
    sdDatalist = None
    sdTIntval = None
    itter= None
    max_bw0 = None
    min_bw0 = None
    n_tollist = None
    # 0 can calculate, 1 can continue ,2 must exit
    checkdata = None
    cache_SDchunk = None
    cache_TDchunk = None
    cache_sortSD = None
    cache_Mlist = None
    cache_MSDchunk = None #only used when cache_type = 3, which is built when cache_type=2
    cache_MTDchunk = None #only used when cache_type = 3, which is built when cache_type=2
    cache_type = None
    if rank==0:  
        sdDatalist,min_bw0,max_bw0 = read(fDatafname)
        sdTIntval = readInteval(fIntervalfname)
        print("sdTIntval",sdTIntval)
        sdTIntval = sdTIntval.reshape((-1,1))
        print("sdTIntval.shape",sdTIntval.shape)
        n_tcnums = sdDatalist[-1].shape[0]
        itter = np.arange(n_tcnums)
        n_tollist = [] 
        for q in sdDatalist:
            n_tollist.append(q.shape[0])     
        checkdata =  check_data_valid(sdDatalist)
        cache_type = 0
        print("Constant:",constant)
    checkdata = comm.bcast(checkdata,root=0)
    if(checkdata == 2):
        sys.exit("MPI Error,Please Check your Input Data")
    n_tollist = comm.bcast(n_tollist,root=0)
    itter = comm.bcast(itter,root=0)
    n_lasttol = len(itter)
    m = int(math.ceil(float(n_lasttol) / size))  
    chunk = itter[rank*m:(rank+1)*m]
    print("chunk are ",chunk)   
    sdDatalist = comm.bcast(sdDatalist,root=0)  
    sdTIntval = comm.bcast(sdTIntval,root=0)
    min_bw0 = comm.bcast(min_bw0,root=0)
    max_bw0 = comm.bcast(max_bw0,root=0) 
    cache_type= comm.bcast(cache_type,root=0) 
    if constant:
        mcal_minbw = np.asarray(sdDatalist[-1][:,3:]).shape[1]+1
    else:
        mcal_minbw = np.asarray(sdDatalist[-1][:,3:]).shape[1]
    nticktime = len(sdDatalist)
    print("mcal_minbw is",mcal_minbw)
    print("nticktime is",nticktime)
    
    X_tollist = [] 
    y_tollist = []
    Coordlist = [] 

    for qt in sdDatalist:
            if constant:
                x_tic = np.hstack([np.ones((qt.shape[0],1)),np.asarray(qt[:,3:])])

            else:
                x_tic = np.asarray(qt[:,3:])              
            X_tollist.append(x_tic)
            y_tollist.append(qt[:,2].reshape((-1,1)))
            Coordlist.append(np.asarray(qt[:,:2]))

    sdDatalist[-1] = sdDatalist[-1][rank*m:(rank+1)*m] 
    print("sdDatalist last is ",sdDatalist[-1].shape[0])
    #begin search and record the calibration time.
    t1 = MPI.Wtime()
    search_fit(min_bw0,max_bw0,sdTIntval)  
    t_last = MPI.Wtime()
    t_duration = t_last-t1
    wt = comm.gather(t_duration, root=0)
    if rank ==0:
        print("Total Time Elapsed:",np.round(max(wt),2),"seconds")
        print("Acturally Time Elapsed:",np.round(t_duration,2),"seconds")
        
    
