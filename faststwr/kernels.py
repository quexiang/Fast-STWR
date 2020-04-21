# STWR and GWR kernel function specifications
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
import scipy
from scipy.spatial.kdtree import KDTree
import numpy as np
from scipy.spatial.distance import cdist as cdist_scipy
from math import radians, sin, cos, sqrt, asin,exp,atan,tan
import copy

def fix_gauss(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed Gaussian kernel.
    """
    w = _Kernel(coords, function='gwr_gaussian', bandwidth=bw,
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_gauss(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive Gaussian kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='gwr_gaussian',
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def fix_bisquare(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed bisquare kernel.
    """
    w = _Kernel(coords, function='bisquare', bandwidth=bw, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_bisquare(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive bisquare kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='bisquare', points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def fix_exp(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed exponential kernel.
    """
    w = _Kernel(coords, function='exponential', bandwidth=bw,
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_exp(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive exponential kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='exponential',
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel
def fix_spt_bisquare(coords_list,y_list,tick_times_intervel,X_list,sita, tick_nums,gwr_bw0,lastcoords =None,
                     points_list=None, alpha =0.3,dspmat=None,dtmat=None,sorted_dmat=None,
                     mbpred = False,spherical=False,build_asp = False,aspList = None,atpList = None,sep_y_star = 0,pred = False,rcdtype = 0):
    """
    Fixed spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,X_list,sita, gwr_bw_list,function='spt_bisquare',lastcoords =lastcoords, 
             points_list=points_list, dspmat_tol=dspmat,dtmat_tol=dtmat,sorted_dmat=sorted_dmat,alpha =alpha,mbpred=mbpred,
             spherical=spherical,build_asp=build_asp,searchfit_asp =aspList,searchfit_atp = atpList,sep_y_star=sep_y_star,pred = pred)
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist#w.d_tmp_list
    elif rcdtype == 2:
        return w.kernel,w.d_spa_list
    elif rcdtype ==3:
        return w.kernel,w.d_spa_list,w.d_tmp_list
    elif rcdtype == 4:
        return w.kernel,w.d_tmp_list
    else:
        return w.kernel
        
def adapt_spt_bisquare(coords_list,y_list,tick_times_intervel,X_list,
                       sita, tick_nums,gwr_bw0,lastcoords =None, 
                       dspal_m_list = None,dsorteds_m_list = None,
                       d_t_list = None,dspmat = None,dtmat=None,
                       points_list=None,alpha =0.3,mbpred = False,spherical=False,build_asp=False,aspList = None,atpList = None,sep_y_star=0,pred = False,rcdtype = 0):
    """
    Spatially adaptive spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,X_list,sita,
                              bk_list=gwr_bw_list, fixed=False,function='spt_bisquare',lastcoords =lastcoords, 
                              dspal_mat_list = dspal_m_list,
                              sorted_dspal_list=dsorteds_m_list,
                              d_tmp_list=d_t_list,
                              dspmat_tol=dspmat,dtmat_tol=dtmat,points_list=points_list,
                              alpha =alpha,mbpred = mbpred,spherical=spherical,build_asp=build_asp,
                              searchfit_asp =aspList,searchfit_atp = atpList,sep_y_star=sep_y_star,pred = pred)#,truncate=False
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist#w.d_tmp_list
    elif rcdtype == 2:
        return w.kernel,w.d_spa_list
    elif rcdtype ==3:
        return w.kernel,w.d_spa_list,w.d_tmp_list
    elif rcdtype == 4:
        return w.kernel,w.d_tmp_list
    else:
        return w.kernel
def fix_spt_gwr_gaussian(coords_list,y_list,tick_times_intervel,X_list,sita, tick_nums,gwr_bw0,lastcoords =None,points_list=None,alpha =0.3,
                         dspmat=None,dtmat=None,sorted_dmat=None,mbpred = False,spherical=False,build_asp=False,aspList= None,atpList = None,sep_y_star=0,pred = False,rcdtype = 0):
    """
    Fixed spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(lastcoords,coords_list, y_list,tick_times_intervel,X_list,sita, gwr_bw_list,function='spt_gwr_gaussian',lastcoords =lastcoords,
             truncate=False,points_list=points_list, dspmat_tol=dspmat,dtmat_tol=dtmat,
             sorted_dmat=sorted_dmat,alpha =alpha,mbpred=mbpred,spherical=spherical,build_asp=build_asp,
             searchfit_asp =aspList,searchfit_atp = atpList,sep_y_star=sep_y_star,pred = pred)
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist#w.d_tmp_list
    elif rcdtype == 2:
        return w.kernel,w.d_spa_list
    elif rcdtype ==3:
        return w.kernel,w.d_spa_list,w.d_tmp_list
    elif rcdtype == 4:
        return w.kernel,w.d_tmp_list
    else:
        return w.kernel

def spt_gwr_gaussian(coords_list,y_list,tick_times_intervel,X_list,
                       sita, tick_nums,gwr_bw0,lastcoords =None,
                       dspal_m_list = None,dsorteds_m_list = None,
                       d_t_list = None,dspmat = None,dtmat=None,points_list=None,alpha =0.3,mbpred = False,
                       spherical=False,build_asp=False,aspList = None,atpList = None,sep_y_star=0,pred = False,rcdtype = 0):
    """
    Spatially adaptive spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,X_list,sita, 
                              bk_list=gwr_bw_list, fixed=False,function='spt_gwr_gaussian',lastcoords =lastcoords,
                              truncate=False,
                              dspal_mat_list = dspal_m_list,
                              sorted_dspal_list=dsorteds_m_list,
                              d_tmp_list=d_t_list,
                              dspmat_tol=dspmat,dtmat_tol=dtmat,points_list=points_list,
                              alpha =alpha,mbpred = mbpred,spherical=spherical,build_asp=build_asp,
                              searchfit_asp =aspList,searchfit_atp = atpList,sep_y_star=sep_y_star,pred =pred)#,truncate=False
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist#w.d_tmp_list
    elif rcdtype == 2:
        return w.kernel,w.d_spa_list
    elif rcdtype ==3:
        return w.kernel,w.d_spa_list,w.d_tmp_list
    elif rcdtype == 4:
        return w.kernel,w.d_tmp_list
    else:
        return w.kernel



from scipy.spatial.distance import cdist

def cdist(coords1,coords2,spherical):
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    n = len(coords1)
    m = len(coords2)
    dmat = np.zeros((n,m))

    if spherical:
        for i in range(n) :
            for j in range(m):
                dmat[i,j] = _haversine(coords1[i][0], coords1[i][1], coords2[j][0], coords2[j][1])
    else:
        dmat = cdist_scipy(coords1,coords2)

    return dmat

class _Kernel(object):
    """
    GWR kernel function specifications.

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=None,
                 function='triangular', eps=1.0000001, ids=None, truncate=True,
                 points=None, dmat=None,sorted_dmat=None, spherical=False): #Added truncate flag
        

        if issubclass(type(data), scipy.spatial.KDTree):
            self.data = data.data
            data = self.data
        else:
            self.data = data
        if k is not None:
            self.k = int(k) + 1
        else:
            self.k = k    
        self.spherical = spherical
        self.searching = True
        
        if dmat is None:
            self.searching = False
        
        if self.searching:
            self.dmat = dmat
            self.sorted_dmat = sorted_dmat
        else:
            if points is None:
                self.dmat = cdist(self.data, self.data, self.spherical)
            else:
                self.points = points
                self.dmat = cdist(self.points, self.data, self.spherical)

        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        self.trunc = truncate
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), 'float') * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()
        self.kernel = self._kernel_funcs(self.dmat/self.bandwidth)
        if self.trunc:
            mask = np.repeat(self.bandwidth, len(self.data), axis=1)
            self.kernel[(self.dmat >= mask)] = 0
                
    def _set_bw(self):
        if self.searching:
            if self.k is not None:
                dmat = self.sorted_dmat[:,:self.k]
            else:
                dmat = self.dmat
        else:
            if self.k is not None:
                dmat = np.sort(self.dmat)[:,:self.k]
            else:
                dmat = self.dmat
        
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(self.data)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return c * np.exp(-(zs ** 2) / 2.)
        elif self.function == 'gwr_gaussian':
            return np.exp(-0.5*(zs)**2)
        elif self.function == 'bisquare':
            return (1-(zs)**2)**2
        elif self.function =='exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)


#计算所有点的权重矩阵，前提是时间带宽len(gwr_bw_list)的期数已知bt_size 等于期数
#注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[0]的距离
#目前只能支持最后一层的采样数据量nsize大于等于之前层采样数据的
#现在构建的权重矩阵是需要将bi_size期所有的cal_data_list2用于t期计算使用
#要有各期的y_value计算出时间权重值所以把y_lsit传入
def cspatiltemporaldist(cal_data_list1,cal_data_list2,y_valuelist,bt_size,deta_t_list,spherical,pred = False):#sita,fname
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    #bt_size = len(gwr_bw_list)-1
    nsize =len(cal_data_list1[-1])
    msize_0 =len(cal_data_list2[-1]) 
#    nlist = [nsize]
    mlist = [msize_0] 
    msize = len(cal_data_list2[-1])
    for p in range(bt_size-1):
        tick_size = len(cal_data_list2[-(p+2)])
        msize += tick_size 
        mlist.append(tick_size)
    dspatialmat_tol = np.zeros((nsize,msize)) 
    detemporalmat_tol = np.zeros((nsize,msize)) 
    dspatialmat  = np.zeros((nsize,msize_0))
    dtemporalmat = np.zeros((nsize,msize_0)) 
    dspatialmat_list = []
    dtemporalmat_list = []
    mb_caltmp = True;
    if bt_size == 1:
        mb_caltmp = False
    if spherical:
        for i in range(nsize) :
            for j in range(msize_0):
                dspatialmat[i,j]  = _haversine(cal_data_list1[-1][i][0], cal_data_list1[-1][i][1], cal_data_list2[-1][j][0], cal_data_list2[-1][j][1])               
    else:
        dspatialmat = cdist_scipy(cal_data_list1[-1],cal_data_list2[-1])#/gwr_bw_list[-1]  

    dspatialmat_tol[0:nsize,0:msize_0] = dspatialmat
#    因为是时间倒序的，dspatialmat_tol[0:nsize,0:nsize]对应的变化是上一期的时间变化矩阵，
#    如果没有上一期的变化矩阵，则置为np.zeros((nsize,nsize)) 
    dspatialmat_list.append(dspatialmat)
    if(mb_caltmp == False):
         detemporalmat_tol[0:nsize,0:msize_0] =dtemporalmat 
         dtemporalmat_list.append(dtemporalmat)  
    elif(pred == False):
        delta_t_total =np.sum(deta_t_list)*1.0
      
        m_size_tick = msize_0       
        dtempfirst = np.zeros((nsize,mlist[0]))
        detemporalmat_tol[:nsize,:mlist[0]]= dtempfirst
        dtemporalmat_list.append(dtempfirst)

        y_value_0 = y_valuelist[-1]
             
        for i in range(bt_size-1): 
            dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
            dtemporalmat_tick = np.zeros((nsize,mlist[i+1]))

            delt_tick_tol = np.sum(deta_t_list[-(2+i):])
            y_value_tick = y_valuelist[-(i+2)]        
            #需要计算对上一期的影响            
            if spherical:
                for j in range(nsize) :
                    for q in range(mlist[i+1]):
                        #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
                        dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])#/gwr_bw_list[-(2+i)]
                        #改造成cal_data_list[-1]点的y与cal_data_list[-2]点y的delt值反应距离
                        dtemporalmat_tick[j,q] =  delta_t_total*abs((y_value_tick[q]-y_value_0[j])/y_value_tick[q])/delt_tick_tol
            else:
                #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
                dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])     
                y_value_tick = y_value_tick.flatten()
                for j in range(nsize) :
                    ydelt_j = np.repeat(y_value_0[j],mlist[i+1],axis=0)
                    dtemporalmat_tick[j] =  delta_t_total*(np.absolute(( y_value_tick- ydelt_j)/y_value_tick))/delt_tick_tol
                                           
            dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick    
            detemporalmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] =dtemporalmat_tick
            dspatialmat_list.append(dspatialmat_tick)
            dtemporalmat_list.append(dtemporalmat_tick)
#            m_size_tick = m_size_tick + mlist[i+1] 
            m_size_tick +=  mlist[i+1] 
    else:#不计算时间，在kernel中重组时间权重矩阵，根据算出的时间矩阵和
         #只计算空间权重返回的时间矩阵无效。
         m_size_tick = msize_0 
         detemporalmat_tol = dtemporalmat
         for i in range(bt_size-1):
             dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
             if spherical:
                for j in range(nsize) :
                    for q in range(mlist[i+1]):
                        #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
                        dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])
             else:
                dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])#/gwr_bw_list[-(2+i)] 
             dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick
             dspatialmat_list.append(dspatialmat_tick)
             m_size_tick = m_size_tick + mlist[i+1] 
    return dspatialmat_list,dtemporalmat_list,dspatialmat_tol,detemporalmat_tol

#只考虑分割计算的情况
def cspatiltemporaldist_spt(cal_data_list1,cal_data_list2,y_valuelist,bt_size,deta_t_list,spherical,
                        sf_asp_dsp = None,sf_atp_dtp = None,y_started =0,pred = False):#sita,fname
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    def _euclidean(Coord_1, Coord_2):
        dist  = sqrt(((Coord_1[0]-Coord_2[0])**2+(Coord_1[1]-Coord_2[1])**2))
        return dist
    nsize =len(cal_data_list1[-1])
    msize_0 =len(cal_data_list2[-1]) 
    mlist = [msize_0] 
    msize = len(cal_data_list2[-1])
    for p in range(bt_size-1):
        tick_size = len(cal_data_list2[-(p+2)])
        msize += tick_size 
        mlist.append(tick_size)
    dspatialmat_tol = np.zeros((nsize,msize)) 
    detemporalmat_tol = np.zeros((nsize,msize))  
    mb_caltmp = True;
    if bt_size == 1:
        mb_caltmp = False        
    if (sf_asp_dsp is not None) & (sf_asp_dsp is not None):  
        dspatialmat_list =sf_asp_dsp
        dtemporalmat_list= sf_atp_dtp
        dspatialmat_tol[0:nsize,0:msize_0] =dspatialmat_list[0]
        detemporalmat_tol[0:nsize,0:msize_0] =dtemporalmat_list[0]
        m_size_tick = msize_0 
        for p in range(bt_size-1):
            dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[p+1]] = dspatialmat_list[p+1]    
            detemporalmat_tol[:nsize,m_size_tick:m_size_tick+mlist[p+1]] =dtemporalmat_list[p+1]  
            m_size_tick += mlist[p+1]
        return dspatialmat_list,dtemporalmat_list,dspatialmat_tol,detemporalmat_tol
    else:      
        dspatialmat  = np.zeros((nsize,msize_0))
        dspatialmat_list = []
        dtemporalmat = np.zeros((nsize,msize_0)) 
        dtemporalmat_list = []          
        if spherical:
                for i in range(nsize) :
                    for j in range(msize_0):
                        dspatialmat[i,j]  = _haversine(cal_data_list1[-1][i][0], cal_data_list1[-1][i][1], cal_data_list2[-1][j][0], cal_data_list2[-1][j][1])#/gwr_bw_list[-1]               
        else:
             dspatialmat = cdist_scipy(cal_data_list1[-1],cal_data_list2[-1])
        dspatialmat_tol[0:nsize,0:msize_0] = dspatialmat
        dspatialmat_list.append(dspatialmat)    
        if(mb_caltmp == False):#只有1期数据
             detemporalmat_tol[0:nsize,0:msize_0] =dtemporalmat 
             dtemporalmat_list.append(dtemporalmat)  
        elif pred == False:#如果不预测，只是计算 
            delta_t_total =np.sum(deta_t_list)*1.0       
            m_size_tick = msize_0       
            dtempfirst = np.zeros((nsize,mlist[0]))
            detemporalmat_tol[:nsize,:mlist[0]]= dtempfirst
            dtemporalmat_list.append(dtempfirst)
#            dtemporalmat_list.append((dtempfirst + m_teps))       
            y_value_0 = y_valuelist[-1]
#            if sf_asp_dsp is None:
            for i in range(bt_size-1):
                dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
                dtemporalmat_tick = np.zeros((nsize,mlist[i+1]))
                delt_tick_tol = np.sum(deta_t_list[-(2+i):])
                y_value_tick = y_valuelist[-(i+2)]
                if spherical:
                    for j in range(nsize) :
                        for q in range(mlist[i+1]):
                            dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])#/gwr_bw_list[-(2+i)]
                            dtemporalmat_tick[j,q] =  delta_t_total*abs((y_value_tick[q]-y_value_0[j+y_started])/y_value_tick[q])/delt_tick_tol                        
                else:
                    #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
                    dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])
                   
                    y_value_tick = y_value_tick.flatten()
                    for j in range(nsize) :
                        ydelt_j = np.repeat(y_value_0[j+y_started],mlist[i+1],axis=0)
                        dtemporalmat_tick[j] =  delta_t_total*(np.absolute(( y_value_tick- ydelt_j)/y_value_tick))/delt_tick_tol
                         
                dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick    
                detemporalmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] =dtemporalmat_tick
                dspatialmat_list.append(dspatialmat_tick)
                dtemporalmat_list.append(dtemporalmat_tick)
                m_size_tick +=  mlist[i+1]

        elif pred == True:#不计算时间，在kernel中重组时间权重矩阵，根据算出的时间矩阵和
             #只计算空间权重返回的时间矩阵无效。
             m_size_tick = msize_0 
             for i in range(bt_size-1):
                 dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
                 if spherical:
                    for j in range(nsize) :
                        for q in range(mlist[i+1]):
                            dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])
                 else:
                    dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])#/gwr_bw_list[-(2+i)] 
                 dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick
                 dspatialmat_list.append(dspatialmat_tick)
                 m_size_tick = m_size_tick + mlist[i+1] 
    
        return dspatialmat_list,dtemporalmat_list,dspatialmat_tol,detemporalmat_tol
 
#注意使用时：要求要对d_tmp进行计算满足：bw_t*abs(delt_v)/delt_t的形式否则应该讲d_tmp赋值为0
def spatialtemporalkernel_funcs(fname,d_spa,d_tmp,m_dtm0,alpha=0.3):
        # functions follow Anselin and Rey (2010) table 5.4
        if m_dtm0:
            if fname == 'spt_triangular':
                return (1 - d_spa)
            elif fname == 'spt_uniform':
                return np.ones(d_spa.shape) * 0.5
            elif fname == 'spt_quadratic':
                return (3. / 4) * (1 - d_spa ** 2)
            elif fname == 'spt_quartic':
                return (15. / 16) * (1 - d_spa ** 2) ** 2
            elif fname == 'spt_gaussian':
                c = np.pi * 2
                c = c ** (-0.5)
                return c * np.exp(-(d_spa ** 2) / 2.)
            elif fname == 'spt_gwr_gaussian':
                return np.exp(-0.5*(d_spa)**2)
            elif fname == 'spt_bisquare':
                return (1-(d_spa)**2)**2
            elif fname =='spt_exponential':
                return np.exp(-d_spa)
            else:
                print('Unsupported kernel function',fname)
        else:
            if fname == 'spt_triangular':
                return ((1-alpha)*(1 - d_spa)+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_uniform':
                return ((1-alpha)*np.ones(d_spa.shape) * 0.5+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_quadratic':
                return ((1-alpha)*(3. / 4) * (1 - d_spa ** 2)+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_quartic':
                return ((1-alpha)*(15. / 16) * (1 - d_spa ** 2) ** 2 + alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_gaussian':
                c = np.pi * 2
                c = c ** (-0.5)
                return ((1-alpha)*(c * np.exp(-(d_spa ** 2) / 2.))+alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_gwr_gaussian':
                return np.exp(-0.5*(d_spa)**2)*(1/(1+np.exp(-d_tmp))-0.5)
            elif fname == 'spt_bisquare':
                return ((1-alpha)*((1-(d_spa)**2)**2)+alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname =='spt_exponential':
                return ((1-alpha)*np.exp(-d_spa)+alpha*(2/(1+np.exp(-d_tmp))-1))
            else:
                print('Unsupported kernel function',fname) 
#注意使用2：曾尝试无用，变化太快。在计算前要对d_tmp进行处理，满足：(bw0-sita*delt_t)/(abs(dsij)+m_eps)形式

class _SpatiotemporalKernel(object):
    """
    GWR Spatiotemporal kernel function specifications.

    """ 
    #knn的思想有可能可以放到时间距离上，做成自适应的。每个回归点对应k期自适应时间距离，用在时间距离矩阵上
    def __init__(self,data_list,y_list,tick_times_intervel,X_list,sita = None, gwr_bw_list = None,#delta_values, 
                 bk_list =None, fixed=True,function='spt_bisquare',eps=1.0000001,lastcoords = None,# ids=None,
                 truncate=True,points_list=None, dspal_mat_list=None,sorted_dspal_list=None,
                 d_tmp_list=None,dspmat_tol=None,dtmat_tol=None,alpha =0.3,mbpred=False,
                 spherical=False,build_asp = False,searchfit_asp = None,searchfit_atp = None,sep_y_star=0,pred = False): #Added truncate flag        
        datalens = len(data_list) 
        if issubclass(type(data_list[0]), scipy.spatial.KDTree):
            for i in range(datalens):
                self.data_list[i] = data_list[i].data
                data_list[i] = self.data_list[i]
        else:
            self.data_list = data_list
        if bk_list is not None:
            self.bk_list = bk_list
            self.nbt_len = len(bk_list)
            for i in range(self.nbt_len):
                self.bk_list[i] = int(bk_list[i]) + 1     
        else:
            self.bk_list = bk_list          
        if gwr_bw_list is not None:
            self.gwr_bw_list = gwr_bw_list
            self.nbt_len = len(gwr_bw_list)
        else:
            self.gwr_bw_list = gwr_bw_list          
        self.y_val_list = y_list[-self.nbt_len:]
        self.tick_times_intls = tick_times_intervel[-self.nbt_len:]
        self.X_list = X_list[-self.nbt_len:]
        self.sita = sita             
        self.spherical = spherical
        self.searching = True         
        self.fname = function.lower()
        self.eps = eps
#        self.m_minbw = data_list[-1].shape[1]+1#最小带宽
        self.m_minbw = X_list[-1].shape[1]+1#最小带宽
        self.Used_meps = True
        m_eps = self.eps-1
        m_dtm0 = False
        self.alpha = alpha
        self.mbpred = mbpred
        self.pred = pred
        self.pre_masktol = None
        self.dst_dtamplist = None #record ==1 used for predit dtamplist 
        
        self.searchfit_asp = searchfit_asp
        self.searchfit_atp = searchfit_atp
        self.sep_y_star=sep_y_star
        #先把datalist中的所有参数保存取来，方便searching使用。    
        nsizes =len(data_list[-1])
        mlist = [nsizes] 
        msizetol = len(data_list[-1])
        for p in range(self.nbt_len-1):
            tick_size = len(data_list[-(p+2)])
            msizetol += tick_size 
            mlist.append(tick_size)
#        n_cur_nsizes = None
        if(self.nbt_len == 1):
            m_dtm0 = True
        if (build_asp  == True):#还没有构建dspmat 
            nsizes = len(lastcoords[-1])
            if self.searchfit_asp is None and self.searchfit_atp is None:
                self.data_list = self.data_list[-self.nbt_len:]
                dspal_mat_list,d_tmp_list,dspmat_tol,dtmat_tol= cspatiltemporaldist_spt(lastcoords, self.data_list,
                                                            self.y_val_list ,self.nbt_len,
                                                            self.tick_times_intls,
                                                            self.spherical,y_started =self.sep_y_star)
                #self.searchfit_asp = dspal_mat_list
            else:#如果有self.searchfit_asp，则直接将其赋给dspal_mat_list,但需要重新计算temporal部分 
                if points_list is None:
                    dspal_mat_list,d_tmp_list,dspmat_tol,dtmat_tol= cspatiltemporaldist_spt(lastcoords, self.data_list,
                                                            self.y_val_list ,self.nbt_len,
                                                            self.tick_times_intls,
                                                            self.spherical,
                                                            sf_asp_dsp = self.searchfit_asp,sf_atp_dtp = self.searchfit_atp )
                else:
                    dspal_mat_list,d_tmp_list,dspmat_tol,dtmat_tol= cspatiltemporaldist_spt(points_list,
                                                                                            self.data_list,
                                                                                            self.y_val_list ,self.nbt_len,
                                                                                            self.tick_times_intls,
                                                                                            self.spherical,
                                                                                            sf_asp_dsp = self.searchfit_asp,sf_atp_dtp = self.searchfit_atp,pred= self.pred)
                
            _len_sorted = len(dspal_mat_list)
            sorted_dspal_list = []
            for _it in range(_len_sorted):
                sorted_dspal_list.append(np.sort(dspal_mat_list[_it]))
        if ((dspmat_tol is None) and (lastcoords is None) or (points_list is not None)):
            self.searching = False
        if self.searching:
            #先构建大矩阵，再搜索时就将大矩阵进行小化，生成搜索需要矩阵形式
            #此时需要将矩阵和参数调整成search时的形式           
            self.d_spa_list = dspal_mat_list[:self.nbt_len]
            self.dsorted_spa =sorted_dspal_list[:self.nbt_len] 
            self.d_tmp_list = d_tmp_list[:self.nbt_len]
#            if(build_asp  == True):
#                nsizes = n_cur_nsizes
            m_curml = mlist[:self.nbt_len]
            m_curtol = sum(m_curml)
            self.dspmat_tol = dspmat_tol[:nsizes,:m_curtol]
        else:
            self.fixed = fixed
            self.trunc = truncate
            #1、准备好cspatiltemporaldist的参数  
            if self.sita is None:
                raise TypeError('Please Enter a sita ', self.sita)
            if points_list is None:
                    if lastcoords is None:
                            self.data_list = self.data_list[-self.nbt_len:]
                            self.d_spa_list, self.d_tmp_list,self.dspmat_tol,self.dtmat_tol = cspatiltemporaldist(self.data_list, self.data_list,self.y_val_list,
                                                                                                                  self.nbt_len,self.tick_times_intls,self.spherical)
                    else:
                        self.data_list = self.data_list[-self.nbt_len:]
                        self.d_spa_list, self.d_tmp_list,self.dspmat_tol,self.dtmat_tol = cspatiltemporaldist(lastcoords, self.data_list,
                                                            self.y_val_list ,self.nbt_len,
                                                            self.tick_times_intls,
                                                            self.spherical)   
                    self.dst_dtamplist = copy.deepcopy(self.d_tmp_list)               
            else:
                self.points_list = points_list[-self.nbt_len:]
#               self.y_val_list --这个应该是最新的y_value_list               
                if(self.pred == False):
                    self.d_spa_list, self.d_tmp_list,self.dspmat_tol,self.dtmat_tol = cspatiltemporaldist(self.points_list, self.data_list, 
                                                         self.y_val_list,self.nbt_len,
                                                         self.tick_times_intls,
                                                         self.spherical,self.pred)
                else:
                    if build_asp == False:#no speated pred
                        d_spa_Pre_list,d_Pre_tmplist,d_Pre_spamat_tol,d_Pre_tmat_tol =cspatiltemporaldist(self.points_list, self.data_list, 
                                                             self.y_val_list,self.nbt_len,
                                                             self.tick_times_intls,
                                                             self.spherical,self.pred) 
                        msizetol = np.sum(mlist)
                        
                        #开始处理预测权重矩阵
                        if m_dtm0 :#如果只使用一期预测则不需要时间权重
                            self.d_spa_list = d_spa_Pre_list
                            self.dspmat_tol = d_Pre_spamat_tol
                            self.d_tmp_list = d_Pre_tmplist
                            self.dtmat_tol  = d_Pre_tmat_tol
                        else:
                            #1找出每个点k个临近点，遍历算出时间权重值，更新时间权重矩阵
                            len_pred = len(self.points_list[0])
    #                        mask_find_neighbors = np.ones_like(d_spa_Pre_list[0])
                            self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                            mask_find_neighbors = d_spa_Pre_list[0].copy()
                            if self.fixed == False:
                                d_sort_spa_tick = d_spa_Pre_list[0].copy()
                                d_sort_spa_tick = np.sort(d_sort_spa_tick)
                                dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]]
                                dspmat_tick = d_sort_spa_tick[:,1:2] 
                                dspmat_last = d_sort_spa_tick[:,-1:]
                                delt_jundge = dspmat_tick == dspmat_last
                                add_neighbor=1
                                while np.any(delt_jundge):
                                    dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]+add_neighbor]
                                    dspmat_tick = dspmat0[:,1:2] 
                                    dspmat_last = dspmat0[:,-1:]
                                    delt_jundge = dspmat_tick == dspmat_last
                                    add_neighbor = add_neighbor+1
                                self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
                                self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
                            if self.trunc:
                                    self.pre_masktol = np.zeros((len_pred,msizetol))
                                    if gwr_bw_list is not None:
                                          self.my_bw_list =self.gwr_bw_list
                                    mask_pre_tick = self.my_bw_list[-1].copy()
                                    mask_pre_tick = np.repeat(mask_pre_tick,mlist[0], axis=1)
                                    #mask_pre_tol[:len_pred,:mlist[0]] = mask_pre_tick
                                    mask_find_neighbors[(mask_find_neighbors>mask_pre_tick)]=0
                                    self.pre_masktol[:,:mlist[0]] =mask_pre_tick
    
                            tol_pre_col = mask_find_neighbors.shape[1]
                            tmp_cal_matrix = d_tmp_list[1].copy() 
                            for cal_timetick in range(self.nbt_len-2):
                                tmp_cal_tick = d_tmp_list[cal_timetick+2].copy()
                                tmp_cal_matrix = np.hstack((tmp_cal_matrix,tmp_cal_tick))
                            d_pre_tmpweight = np.zeros((len_pred,msizetol-mlist[0]))
                            for Pre_row in  range(len_pred):
    #                            mask_find_neighbors[Pre_row]
    #                            neighbor_elemets = []
                                compress_matrix = []
                                #previous coord
                                prev_distances = []
                                for Pre_col in range(tol_pre_col):
                                    if(mask_find_neighbors[Pre_row,Pre_col]>0):
                                        #Pre_col
                                        no_zero_val = d_Pre_spamat_tol[Pre_row,Pre_col+mlist[0]] 
                                        if no_zero_val == 0:
                                              no_zero_val = d_Pre_spamat_tol[Pre_row,Pre_col+mlist[0]]+m_eps              
                                        pv_dist = 1.0/no_zero_val
#                                        pv_dist = 1.0/pv_dist
                                        prev_distances.append(pv_dist)
                                        compress_matrix.append(pv_dist*tmp_cal_matrix[Pre_col])
                                compress_matrix = np.asarray(compress_matrix)
                                tol_distances = np.sum(prev_distances)
                                compress_matrix= compress_matrix/tol_distances
                                d_pre_tmpweight[Pre_row] = np.sum(compress_matrix, axis=0)
                                #直接求简单平均
#                                d_pre_tmpweight[Pre_row] = np.mean(compress_matrix, axis=0)
                                #距离反比加权法
                                
                                
                            self.d_spa_list = d_spa_Pre_list
                            self.dspmat_tol = d_Pre_spamat_tol
                            
                            self.d_tmp_list = d_Pre_tmplist
                            d_Pre_tmat_tol = np.hstack((d_Pre_tmat_tol,d_pre_tmpweight))
                            self.dtmat_tol = d_Pre_tmat_tol

                    else: #spearted Pred

                      
                            msizetol = np.sum(mlist) #len(self.data_list[-1])
                 
                            #开始处理预测权重矩阵
                            if m_dtm0 :#如果只使用一期预测则不需要时间权重
                                self.d_spa_list = dspal_mat_list
                                self.dspmat_tol = dspmat_tol
                                self.d_tmp_list = d_tmp_list
                                self.dtmat_tol  = dtmat_tol
                            else:
                                #1找出每个点k个临近点，遍历算出时间权重值，更新时间权重矩阵
                                len_pred = len(self.points_list[0])
                                self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                                mask_find_neighbors = dspal_mat_list[0].copy()
                                if self.fixed == False:
                                    d_sort_spa_tick = dspal_mat_list[0].copy()
                                    d_sort_spa_tick = np.sort(d_sort_spa_tick)
                                    dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]]
                                    dspmat_tick = d_sort_spa_tick[:,1:2] 
                                    dspmat_last = d_sort_spa_tick[:,-1:]
                                    delt_jundge = dspmat_tick == dspmat_last
                                    add_neighbor=1
                                    while np.any(delt_jundge):
                                        dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]+add_neighbor]
                                        dspmat_tick = dspmat0[:,1:2] 
                                        dspmat_last = dspmat0[:,-1:]
                                        delt_jundge = dspmat_tick == dspmat_last
                                        add_neighbor = add_neighbor+1
                                    self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
                                    self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
                                if self.trunc:
                                        self.pre_masktol = np.zeros((len_pred,msizetol))
                                        if gwr_bw_list is not None:
                                              self.my_bw_list =self.gwr_bw_list
                                        mask_pre_tick = self.my_bw_list[-1].copy()
                                        mask_pre_tick = np.repeat(mask_pre_tick,mlist[0], axis=1)
                                        #mask_pre_tol[:len_pred,:mlist[0]] = mask_pre_tick
                                        mask_find_neighbors[(mask_find_neighbors>mask_pre_tick)]=0
                                        self.pre_masktol[:,:mlist[0]] =mask_pre_tick
        
                                tol_pre_col = mask_find_neighbors.shape[1]
                                d_pre_tmpweight = np.zeros((len_pred,msizetol-mlist[0]))
                                
#                                d_tmp_list = self.searchfit_atp[0]
##                                    tmp_cal_matrix = d_tmp_list[1].copy()
#                                tmp_cal_matrix = d_tmp_list[1]
#                                for cal_timetick in range(self.nbt_len-2):
#                                    tmp_cal_tick = d_tmp_list[cal_timetick+2].copy()
#                                    tmp_cal_matrix = np.hstack((tmp_cal_matrix,tmp_cal_tick))
#                                for Pre_row in  range(len_pred):
#                                    compress_matrix = []
#                                    for Pre_col in range(tol_pre_col):
#                                        if(mask_find_neighbors[Pre_row,Pre_col]>0):
#                                            compress_matrix.append(tmp_cal_matrix[Pre_col])
#                                    compress_matrix = np.asarray(compress_matrix)
#                                    d_pre_tmpweight[Pre_row] = np.mean(compress_matrix, axis=0)     
                                tol_separts = len(self.searchfit_atp)
                                for Pre_row in  range(len_pred):
                                    compress_matrix = []
                                    start_idx = 0
                                    for sep_itr in range(tol_separts):
                                        d_tmp_list = self.searchfit_atp[sep_itr]
                                        tmp_cal_matrix = d_tmp_list[1]
                                        len_curtic = tmp_cal_matrix.shape[0]
                                        for cal_timetick in range(self.nbt_len-2):
                                            tmp_cal_tick = d_tmp_list[cal_timetick+2].copy()
                                            tmp_cal_matrix = np.hstack((tmp_cal_matrix,tmp_cal_tick))
                                            
                                        for Pre_col in range(len_curtic):
                                            if(mask_find_neighbors[Pre_row,start_idx+Pre_col]>0):
                                                            compress_matrix.append(tmp_cal_matrix[Pre_col])
                                        start_idx += len_curtic
                                    compress_matrix = np.asarray(compress_matrix)
                                    d_pre_tmpweight[Pre_row] = np.mean(compress_matrix, axis=0)
   
                                self.d_spa_list = dspal_mat_list
                                self.dspmat_tol = dspmat_tol
                                #self.d_tmp_list = d_Pre_tmplist
                                self.dtmat_tol = np.hstack((dtmat_tol,d_pre_tmpweight))
                        
        if self.pred:
                 self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                 self._set_spt_sita()
                 self.kernel = np.zeros(self.dspmat_tol.shape) 
                 len_pred = len(self.points_list[0])            
                 cal_sptol = self.d_spa_list[0]
                 if m_dtm0:
                    cal_sptol=cal_sptol/self.my_bw_list[-1]
                    cal_my_ttol = self.d_tmp_list[0]
                    self.kernel[:,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha)
                 else:
                    m_bwcop =  self.my_bw_list[-1].copy()
                    cal_sptol = cal_sptol/m_bwcop
                    cal_my_ttol = self.dtmat_tol[:,:mlist[0]]
                    if self.trunc:
                        cal_sptol[cal_sptol>=1]=1 
                        cal_my_ttol[cal_sptol>=1] =0 
#                    self.kernel[:,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha) 
                    self.kernel[:,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,True,self.alpha) 
                 
                 #将整个pre_masktol 和 kernel构建出来
                 m_size_tick = mlist[0]
                 for i in range(self.nbt_len-1):
                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
                     bw_expand =self.my_bw_list[-(2+i)].copy()
                     d_sort_spa_tick = self.d_spa_list[i+1].copy()
                     d_sort_spa_tick = np.sort(d_sort_spa_tick)
                     sorted_dspmat_min = d_sort_spa_tick[:,:self.m_minbw]
                     dspmat_tick = sorted_dspmat_min[:,1:2]
                     dspmat_last = sorted_dspmat_min[:,-1:]
                     delt_jundge_same = dspmat_tick == dspmat_last
                     add_neighbor=1
                     while np.any(delt_jundge_same):
                          sorted_dspmat_min = d_sort_spa_tick[:,:self.m_minbw+add_neighbor]
                          dspmat_tick = sorted_dspmat_min[:,1:2] 
                          dspmat_last = sorted_dspmat_min[:,-1:]
                          delt_jundge_same = dspmat_tick == dspmat_last
                          add_neighbor = add_neighbor+1 
                     cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                     delt_jundge = bw_expand < cal_bw_min
                     if np.any(delt_jundge):
                         np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge) 
                     cal_sptol_tick = self.d_spa_list[i+1].copy()
                     cal_tmp_tick = self.dtmat_tol[:,m_size_tick:(m_size_tick+mlist[i+1])]
                     if self.trunc:
                         mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
                         cal_sptol_tick = cal_sptol_tick/bw_expand
                         cal_sptol_tick[cal_sptol_tick>=1]=1
                         cal_tmp_tick[cal_sptol_tick>=1] =0 
                         self.pre_masktol[:,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick
                     else:
                         cal_sptol_tick = cal_sptol_tick/bw_expand            
#                     if self.trunc:
#                         cal_sptol_tick[cal_sptol_tick>=1]=1 
                        
                     self.kernel[:,m_size_tick:(m_size_tick+mlist[i+1])]=spatialtemporalkernel_funcs(self.fname,
                             cal_sptol_tick,cal_tmp_tick,m_dtm0,self.alpha)
                     m_size_tick +=mlist[i+1]
                 if self.pre_masktol is not None:
                     self.kernel[(self.dspmat_tol > self.pre_masktol)] = 0
                 #avoid singular matrix
                 self.kernel += np.random.normal(0,m_eps,self.kernel.shape)
        else:                        
            self.fixed = fixed
            self.trunc = truncate     
            nsizes = self.d_spa_list[0].shape[0]
            #相当于用户指定了带宽，则直接构造出各层的mask
            if gwr_bw_list is not None:
                try:
                    self.gwr_bw_list[-1] = np.array(self.gwr_bw_list[-1])
                    self.gwr_bw_list[-1].shape = (len(self.gwr_bw_list[-1]),1)
                except:
                    self.gwr_bw_list[-1] = np.ones((len(data_list[-1]),1),'float')*self.gwr_bw_list[-1] 
                for i in range(self.nbt_len-1):
                    self.gwr_bw_list[-(2+i)] =  self.gwr_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
                    try:
                        self.gwr_bw_list[-(2+i)] = np.array(self.gwr_bw_list[-(2+i)])
                        self.gwr_bw_list[-(2+i)].shape = (len(self.gwr_bw_list[-(2+i)]), 1)
                    except:
                        self.gwr_bw_list[-(2+i)] = np.ones((len(data_list[-(2+i)]), 1), 'float') * self.gwr_bw_list[-(2+i)]               
            else:
                     self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                     self._set_spt_sita()
            #先构建mask裁剪空间距离和XList之间的关系,判断是否有解，若有解则后面不需要再构建dspatmat，直接使用即可
            mask_tol = np.zeros((nsizes,msizetol))
            if self.trunc:
                    if gwr_bw_list is not None:
                          self.my_bw_list =self.gwr_bw_list
#                    bw_expandlist = [self.my_bw_list[-1]]
                    mask_tick = self.my_bw_list[-1].copy()
                    mask_tick = np.repeat(mask_tick,mlist[0], axis=1)
                    mask_tol[:nsizes,:mlist[0]] = mask_tick
                    cal_sptol = self.d_spa_list[0].copy()
                    cal_sptol = cal_sptol/self.my_bw_list[-1] 
                    cal_sptol[cal_sptol>=1]=1 
                    self.d_tmp_list[0][cal_sptol>=1] =0 
                    cal_sptol += np.absolute(np.random.normal(0,m_eps,cal_sptol.shape))
                    self.d_tmp_list[0]+=np.absolute(np.random.normal(0,m_eps,cal_sptol.shape))
                    cal_sptol_list= [cal_sptol]
                    if self.Used_meps == False:
                        s_xtol = self.X_list[-1].shape[1]
                        jd_x_tol = np.zeros((cal_sptol.shape[0],s_xtol))
                        for x_iter in range(s_xtol):
                            jd_x_cal_ti = self.X_list[-1][:,x_iter:x_iter+1].copy()
                            jd_x_cal_ti_T = jd_x_cal_ti.T
                            jd_x_cal_ti_T = np.repeat(jd_x_cal_ti_T,cal_sptol.shape[0], axis=0)
                            jd_x_tol_tic = jd_x_cal_ti_T*cal_sptol
                            jd_x_tol_tic = np.sum(jd_x_tol_tic,axis=1)
                            jd_x_tol[:,x_iter:x_iter+1] += jd_x_tol_tic.reshape((-1,1))
                      
                        m_size_tick = mlist[0]
                        for i in range(self.nbt_len-1):
                            if gwr_bw_list is not None:
                                self.my_bw_list =self.gwr_bw_list  
                            self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
                            bw_expand =self.my_bw_list[-(2+i)].copy()
                            if self.searching:
                                 sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw]
                                 dspmat_tick = sorted_dspmat_min[:,1:2]
                                 dspmat_last = sorted_dspmat_min[:,-1:]
                                 delt_jundge_same = dspmat_tick == dspmat_last
                                 add_neighbor=1
                                 while np.any(delt_jundge_same):
                                      sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw+add_neighbor]
                                      dspmat_tick = sorted_dspmat_min[:,1:2] 
                                      dspmat_last = sorted_dspmat_min[:,-1:]
                                      delt_jundge_same = dspmat_tick == dspmat_last
                                      add_neighbor = add_neighbor+1 
                                 cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                                 delt_jundge = bw_expand < cal_bw_min
                                 if np.any(delt_jundge):
                                     np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge)
    #                        bw_expandlist.append(bw_expand)
                            mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
                            mask_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick
                            
                            cal_sptol_tick = self.d_spa_list[i+1].copy()
                            cal_sptol_tick = cal_sptol_tick/bw_expand
                            cal_sptol_tick[cal_sptol_tick>=1]=1 
                            self.d_tmp_list[i+1][cal_sptol_tick>=1] =0 
                            cal_sptol_tick += np.absolute(np.random.normal(0,m_eps,cal_sptol_tick.shape))
                            self.d_tmp_list[i+1] += np.absolute(np.random.normal(0,m_eps,cal_sptol_tick.shape))
                            
                            cal_sptol_tick += np.absolute(np.random.normal(0,m_eps,cal_sptol_tick.shape))
    #                        cal_sptol_tick = cal_sptol_tick/bw_expand
                            cal_sptol_list.append(cal_sptol_tick)
                            
                            for x_iter in range(s_xtol):
                                jd_x_cal_ti = self.X_list[-(i+2)][:,x_iter:x_iter+1].copy()
                                jd_x_cal_ti_T = jd_x_cal_ti.T
                                jd_x_cal_ti_T = np.repeat(jd_x_cal_ti_T,cal_sptol_tick.shape[0], axis=0)
                                jd_x_tol_tic = jd_x_cal_ti_T*cal_sptol_tick
                                jd_x_tol_tic = np.sum(jd_x_tol_tic,axis=1)
                                jd_x_tol[:,x_iter:x_iter+1] += jd_x_tol_tic.reshape((-1,1))
    
                            m_size_tick +=mlist[i+1]
                        x_s_jundge = jd_x_tol == 0
                        #否则raise error （为了减少搜索带宽的时间）
                        if np.any(x_s_jundge):
                             raise  Exception('Singular matrix')
                    else:#self.Used_meps == True (default)
                        m_size_tick = mlist[0]
                        for i in range(self.nbt_len-1):
                            if gwr_bw_list is not None:
                                self.my_bw_list =self.gwr_bw_list  
                            self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
                            bw_expand =self.my_bw_list[-(2+i)].copy()
                            if self.searching:
                                 sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw]
                                 dspmat_tick = sorted_dspmat_min[:,1:2]
                                 dspmat_last = sorted_dspmat_min[:,-1:]
                                 delt_jundge_same = dspmat_tick == dspmat_last
                                 add_neighbor=1
                                 while np.any(delt_jundge_same):
                                      sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw+add_neighbor]
                                      dspmat_tick = sorted_dspmat_min[:,1:2] 
                                      dspmat_last = sorted_dspmat_min[:,-1:]
                                      delt_jundge_same = dspmat_tick == dspmat_last
                                      add_neighbor = add_neighbor+1 
                                 cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                                 delt_jundge = bw_expand < cal_bw_min
                                 if np.any(delt_jundge):
                                     np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge)
                            mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
                            mask_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick                        
                            cal_sptol_tick = self.d_spa_list[i+1].copy()
                            cal_sptol_tick = cal_sptol_tick/bw_expand
                            cal_sptol_tick[cal_sptol_tick>=1]=1 
                            self.d_tmp_list[i+1][cal_sptol_tick>=1] =0 
                            cal_sptol_tick += np.absolute(np.random.normal(0,m_eps,cal_sptol_tick.shape))
                            self.d_tmp_list[i+1] += np.absolute(np.random.normal(0,m_eps,cal_sptol_tick.shape))
                            
                            cal_sptol_list.append(cal_sptol_tick) 
            self.kernel = np.zeros(self.dspmat_tol.shape)       
            #需要先空间trunc掉，避免没有trunc就参与计算第一层的权重矩阵
#            if self.trunc:
##                    if gwr_bw_list is not None:
##                          self.my_bw_list =self.gwr_bw_list
##                    mask_tick = self.my_bw_list[-1].copy()
##                    mask_tick = np.repeat(mask_tick,mlist[0], axis=1)
##                    mask_tol[:nsizes,:mlist[0]] = mask_tick
#                    mask_tick = mask_tol[:nsizes,:mlist[0]]  
#                    cal_sptol[(cal_sptol>mask_tick)]=0
#            if m_dtm0:
#                cal_sptol=cal_sptol/self.my_bw_list[-1]
            if self.trunc:
                cal_sptol = cal_sptol_list[0]
            else:
                cal_sptol = self.d_spa_list[0]
            cal_my_ttol = self.d_tmp_list[0]
#            self.kernel[:nsizes,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha)
            self.kernel[:nsizes,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,True,self.alpha)
#            else:
#                m_bwcop =  self.my_bw_list[-1].copy()
#                #cal_sptol = abs(cal_sptol) + m_eps
#                #m_bwcop = np.repeat(m_bwcop,nsizes,axis=1)
#                cal_sptol = cal_sptol/m_bwcop
#                cal_sptol = cal_sptol/bw_expandlist[0]
#                cal_my_ttol = self.d_tmp_list[0]
#                self.kernel[:nsizes,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha) 
            m_size_tick = mlist[0]   
            for i in range(self.nbt_len-1):
#                     if gwr_bw_list is not None:
#                         self.my_bw_list =self.gwr_bw_list  
#                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
#                     bw_expand =self.my_bw_list[-(2+i)].copy()
#                     if self.searching:
#                         sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw]
#                         dspmat_tick = sorted_dspmat_min[:,1:2]
#                         dspmat_last = sorted_dspmat_min[:,-1:]
#                         delt_jundge_same = dspmat_tick == dspmat_last
#                         add_neighbor=1
#                         while np.any(delt_jundge_same):
#                              sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw+add_neighbor]
#                              dspmat_tick = sorted_dspmat_min[:,1:2] 
#                              dspmat_last = sorted_dspmat_min[:,-1:]
#                              delt_jundge_same = dspmat_tick == dspmat_last
#                              add_neighbor = add_neighbor+1 
#                         cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
#                         delt_jundge = bw_expand < cal_bw_min
#                         if np.any(delt_jundge):
#                             np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge)                        
#                     spa_mat_tick= self.d_spa_list[i+1]
#                     #spa_mat_cop = spa_mat_tick.copy()
#                     cal_sptol_tick = spa_mat_tick.copy()
#                     self.dspmat_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=spa_mat_tick
#                     if self.trunc:
#                         mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
#                         mask_tick = mask_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]
#                         cal_sptol_tick[(cal_sptol_tick>mask_tick)]=0
#                         mask_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick
                     
                     #spa_mat_cop = abs(spa_mat_cop)+m_eps
                     #mbw0_sita_dt = np.repeat(bw_expand,nsizes, axis=1)
                     # cal_sptol_tick = mbw0_sita_dt/spa_mat_cop
#                     cal_sptol_tick= cal_sptol_tick/bw_expand
#                         cal_sptol_tick= cal_sptol_tick/bw_expandlist[i+1]
                     
                     if self.trunc:
                         cal_sptol_tick = cal_sptol_list[i+1]
                     else:
                         cal_sptol_tick = self.d_spa_list[i+1]
                     self.kernel[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=spatialtemporalkernel_funcs(self.fname,
                             cal_sptol_tick,self.d_tmp_list[i+1],m_dtm0,self.alpha)
                     m_size_tick +=mlist[i+1]
    #因为在spatialtemporal_funcs时，核函数除了按期对角以外的元素为0，参与计算后由于exp 0 =1,所以要消除了对角块以外的1             
            #还需要在mask一下：
#            self.kernel[(self.dspmat_tol > mask_tol)] = 0
            #avoid singular matrix
#            self.kernel += np.random.normal(0,m_eps,self.kernel.shape)


#        if self.trunc:
#        #需要将每个时间的maks合并成大mask
#            if gwr_bw_list is not None:
#                  self.my_bw_list =self.gwr_bw_list
#            nsizes = len(self.data_list[-1])
#            mask_tol = np.zeros((self.nbt_len*nsizes,self.nbt_len*nsizes)) 
#            mask_tick = np.repeat(self.my_bw_list[-1],nsizes,axis=1)
#            mask_tol[:nsizes,:nsizes] = mask_tick
#            cal_sptol = self.dspmat_tol.copy()
#            
#            for i in range(self.nbt_len-1):
#                nsize_tick = len(self.data_list[-(2+i)])
#                self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
#                
##                bw_expand = self.my_bw_list[-(2+i)].copy()
##                cal_sptol_tick = cal_sptol[(i+1)*nsizes:(i+2)*nsizes,(i+1)*nsizes:(i+2)*nsizes]
##                cal_sptol_tmin = np.min(cal_sptol_tick,axis =1)
##                for j in range(nsize_tick):
##                     if(abs(cal_sptol_tmin[j]) > abs(bw_expand[j])):
##                              bw_expand[j] = cal_sptol_tmin[j]*self.eps
#                                             
##                need_expanded = np.abs(kernel_minx).reshape(-1,1) - np.abs(bw_expand).reshape(-1,1)
##                need_expanded_len=len(need_expanded) 
##                for j in range(need_expanded_len):
#                #if(self.my_bw_list[-(2+i)])
##                mask_tick = np.repeat(bw_expand,nsize_tick, axis=1)
#                mask_tick = np.repeat(self.my_bw_list[-(2+i)],nsize_tick, axis=1)
#                #masklist = np.hstack((masklist,mask_tick))
#                mask_tol[(i+1)*nsizes:(i+1)*nsizes+nsize_tick,(i+1)*nsizes:(i+1)*nsizes+nsize_tick]=mask_tick
#            self.kernel[(np.abs(self.dspmat_tol) > np.abs(mask_tol))] = 0

        # 主要是计算k个近邻的距离。因为每个回归点的初始距离不知道。
        #注意这里只能设置最近一层带宽，然后再根据各回归点的带宽计算出相应时间空间点的带宽            
    def _set_spt_sita(self):
            if self.searching:
                if self.bk_list is not None:
                    dspmat0 =self.dsorted_spa[0][:,:self.bk_list[-1]]
                    dspmat_tick = dspmat0[:,1:2] 
                    dspmat_last = dspmat0[:,-1:]
                    delt_jundge = dspmat_tick == dspmat_last
                    add_neighbor=1
                    while np.any(delt_jundge):
                        dspmat0 =self.dsorted_spa[0][:,:self.bk_list[-1]+add_neighbor]
                        dspmat_tick = dspmat0[:,1:2] 
                        dspmat_last = dspmat0[:,-1:]
                        delt_jundge = dspmat_tick == dspmat_last
                        add_neighbor = add_neighbor+1
#                         np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge)
                else:    
                    dspmat0 = self.d_spa_list[0] 
            else:
                if self.bk_list is not None:
                    dspmat0  = np.sort(self.d_spa_list[0])[:,:self.bk_list[-1]]
                    #放到后面再sort一下其他各层，因为需要bandwidth向量
                else:
                    dspmat0 =  self.d_spa_list[0] 
            if self.fixed:
                # use max knn distance as bandwidth
                bandwidth0 = dspmat0.max() * self.eps
                n = len(self.data_list[-1])
                self.my_bw_list[-1] = np.ones((n, 1), 'float') * bandwidth0
                #依次计算各层带宽根据sita来算,放到算mask的时候在算                
            else:
                # use local max knn distance
                self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
                self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
                
                                        
   
        #1、要每层的points数量去构建权重矩阵      
        #2、需要先构建所有数据data_list[-1].size *(data_list[-1]+data_list[-2]+.....data_list[0])的dmat
        #3、利用各层的bandwidth去盖住每个点距离计算点超过时空带宽的bandwidth的权值，超过bandwidth的为0
#                        
#                    self.bws_list[-1] =bk_list[-1]                    
#                    dsptmat = self.sorted_dmat[:,0:self.bws_list[-1]]
#                    start_cur= self.bws_list[-1]
#                    for i in range(self.nbt_len-1):
#                         self.bws_list[-(2+i)] = int(self.bws_list[-(i+1)] - sita*self.delta_vals[-(i+1)])  
#                         dmat_tick  = self.sorted_dmat[:,start_cur:start_cur+self.bws_list[-(2+i)]]
#                         start_cur += self.bws_list[-(2+i)]
#                         dsptmat = np.hstack((dsptmat,dmat_tick))
#                else:
#                    dsptmat = self.dspt_mat
#            else:
#                if bk_list is not None:
#                    self.bws_list[-1] =bk_list[-1]
#                    dsptmat = self.dspt_mat[:,0:self.bws_list[-1]]
#                    dsptmat = np.sort(dsptmat)[:,:self.bws_list[-1]]                
#                    start_cur= self.bws_list[-1]
#                    for i in range(self.nbt_len):
#                         self.bws_list[-(2+i)] = int(self.bws_list[-(i+1)] - sita*self.delta_vals[-(i+1)])  
#                         dmat_tick  = self.dspt_mat[:,start_cur:start_cur+self.bws_list[-(2+i)]]
#                         dmat_tick =  np.sort(dsptmat)[0,:self.bws_list[-(2+i)]]
#                         start_cur += self.bws_list[-(2+i)]
#                         dsptmat = np.hstack((dsptmat,dmat_tick))
#                else:
#                    dsptmat = self.dspt_mat 
#            #这以下暂时没用
#            if self.fixed:
#                # use max knn distance as bandwidth                    
#                bsmax = dsptmat.max() * self.eps               
#                n = len(self.data_list[-1])
#                self.bsmax = np.ones((n, 1), 'float') * bsmax
#            else:
#                # use local max knn distance
#                self.bsmax = dsptmat.max(axis=1) * self.eps
#                self.bsmax.shape = (self.bsmax.size, 1)           
       
             
##只build空间的
#def cspatildist(cal_data_list1,cal_data_list2,y_valuelist,bt_size,deta_t_list,spherical,build_asp = False,sf_asp_dsp = None,pred = False):
#    def _haversine(lon1, lat1, lon2, lat2):
#        R = 6371.0 # Earth radius in kilometers
#        dLat = radians(lat2 - lat1)
#        dLon = radians(lon2 - lon1)
#        lat1 = radians(lat1)
#        lat2 = radians(lat2)
#        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
#        c = 2*asin(sqrt(a))
#        return R * c
#    def _euclidean(Coord_1, Coord_2):
#        dist  = sqrt(((Coord_1[0]-Coord_2[0])**2+(Coord_1[1]-Coord_2[1])**2))
#        return dist
#    nsize =len(cal_data_list1[-1])
#    msize_0 =len(cal_data_list2[-1]) 
#
#    mlist = [msize_0] 
#    msize = len(cal_data_list2[-1])
#    for p in range(bt_size-1):
#        tick_size = len(cal_data_list2[-(p+2)])
#        msize += tick_size 
#        mlist.append(tick_size)
#    dspatialmat_tol = np.zeros((nsize,msize)) 
#    if sf_asp_dsp is None :
#        dspatialmat  = np.zeros((nsize,msize_0))
#        dspatialmat_list = []
#    else:
#        dspatialmat_list =sf_asp_dsp 
#    mb_caltmp = True;
#    if bt_size == 1:
#        mb_caltmp = False
#    if spherical:
#        if (sf_asp_dsp is None):
#            for i in range(nsize) :
#                for j in range(msize_0):
#                    dspatialmat[i,j]  = _haversine(cal_data_list1[-1][i][0], cal_data_list1[-1][i][1], cal_data_list2[-1][j][0], cal_data_list2[-1][j][1])#/gwr_bw_list[-1]                 
#    else:
#        if build_asp:
#            if sf_asp_dsp is None:
#                #先用cdist_scipy算一部分，剩下的在用_euclidean算
#                n_cal_tic =nsize 
#                m_cal_tic =msize_0
#                mat_tic = 0
#                while(n_cal_tic<m_cal_tic):
#                    dspatialmat[:,nsize*mat_tic:nsize*(mat_tic+1)] = cdist_scipy(cal_data_list1[-1],
#                               cal_data_list2[-1][nsize*mat_tic:nsize*(mat_tic+1),:])
#                    mat_tic +=1
#                    n_cal_tic += nsize
#                m_cal_start =  mat_tic*nsize
#                m_cal_tic = msize_0 -m_cal_start
#                #计算剩余的,还可以在进一步优化加快速度
#                #calculate seprate Distance 
#                for i in range(nsize):
#                    for j in range(m_cal_tic):
#                         dspatialmat[i,j+m_cal_start] =_euclidean(cal_data_list1[-1][i],cal_data_list2[-1][j+m_cal_start]) 
#        else:
#            dspatialmat = cdist_scipy(cal_data_list1[-1],cal_data_list2[-1])#/gwr_bw_list[-1]  
#    if sf_asp_dsp is None:
#        dspatialmat_tol[0:nsize,0:msize_0] = dspatialmat
#        dspatialmat_list.append(dspatialmat)
#    else:
#        dspatialmat_tol[0:nsize,0:msize_0] =dspatialmat_list[0] 
#    if pred == False:
#        delta_t_total =np.sum(deta_t_list)*1.0      
#        m_size_tick = msize_0        
#      
#        if sf_asp_dsp is None:
#            for i in range(bt_size-1):
#
#                dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
#            
#    
#                delt_tick_tol = np.sum(deta_t_list[-(2+i):])
#
#                y_value_tick = y_valuelist[-(i+2)]
#      
#                if spherical:
#                    for j in range(nsize) :
#                        for q in range(mlist[i+1]):
#                            #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
#                            dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])#/gwr_bw_list[-(2+i)]
#                            #改造成cal_data_list[-1]点的y与cal_data_list[-2]点y的delt值反应距离                           
#                else:
#                    #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
#                    dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])#/gwr_bw_list[-(2+i)] 
#                dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick 
#                dspatialmat_list.append(dspatialmat_tick)
#                m_size_tick +=  mlist[i+1]
#        else:#sf_asp_dsp
#            for i in range(bt_size-1):
#                dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = sf_asp_dsp[i+1] 
#                m_size_tick +=  mlist[i+1]    
#    elif(pred == True)&(sf_asp_dsp is None):#不计算时间，在kernel中重组时间权重矩阵，根据算出的时间矩阵和
#         #只计算空间权重返回的时间矩阵无效。
#         m_size_tick = msize_0 
#         for i in range(bt_size-1):
#             dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
#             if spherical:
#                for j in range(nsize) :
#                    for q in range(mlist[i+1]):
#                        #注意：空间距离是指cal_data_list2数据点第二层开始到cal_data_list1[-1]的距离
#                        dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])
#             else:
#                dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])#/gwr_bw_list[-(2+i)] 
#             dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick
#             dspatialmat_list.append(dspatialmat_tick)
#             m_size_tick = m_size_tick + mlist[i+1] 
#
#    return dspatialmat_list,dspatialmat_tol
   
##只build时间的，空间不管
#def ctemporaldist(cal_data_list1,cal_data_list2,y_valuelist,bt_size,deta_t_list,spherical,build_atp = False,sf_atp_dtp = None,pred = False):#sita,fname
 
           


##########优化计算时间距离或球面距离
#                n_y_tick = y_value_tick.shape[0]
#                if(n_y_v0 >= n_y_tick ):
#                    mat_tic_tick = 0
#                    cal_y0_tick = n_y_tick
#                    while(cal_y0_tick <= n_y_v0):
#                        for s_j in range(nsize):
#                            dtemporalmat_tick[s_j] = delta_t_total*(abs(y_value_tick[:]-y_value_0[n_y_tick*mat_tic_tick:n_y_tick*(mat_tic_tick+1)])/y_value_tick[:])/delt_tick_tol
#                        mat_tic_tick += 1
#                        cal_y0_tick += n_y_tick
#                    m_y_cal_start =  mat_tic_tick*n_y_tick
#                    cal_y0_tick = n_y_v0 -m_y_cal_start
#                    for s_j in range(nsize):
#                        for s_yiter in range(cal_y0_tick):
#                            dtemporalmat_tick[s_j,m_y_cal_start+s_yiter] =  delta_t_total*(abs(y_value_tick[m_y_cal_start+s_yiter]-y_value_0[s_j])/y_value_tick[m_y_cal_start+s_yiter])/delt_tick_tol
#                else: #n_y_v0<n_y_tick
#                    mat_tic_tick = 0
#                    cal_y_tick = n_y_v0
#                    while(cal_y_tick < n_y_tick):
#                        for s_j in range(nsize):
#                            sssss= delta_t_total*(abs(y_value_tick[mat_tic_tick*n_y_v0:(mat_tic_tick+1)*n_y_v0]-y_value_0[:])/y_value_tick[mat_tic_tick*n_y_v0:(mat_tic_tick+1)*n_y_v0])/delt_tick_tol
#                            dtemporalmat_tick[s_j][mat_tic_tick*n_y_v0:(mat_tic_tick+1)*n_y_v0]  = sssss.flatten()
#                        mat_tic_tick += 1
#                        cal_y_tick += n_y_v0
#                    m_y_cal_start =  mat_tic_tick*n_y_v0
#                    cal_y0_tick = n_y_tick -m_y_cal_start
#                    for s_j in range(nsize):
#                        for s_yiter in range(cal_y0_tick):
#                            dtemporalmat_tick[s_j,m_y_cal_start+s_yiter] =  delta_t_total*(abs(y_value_tick[m_y_cal_start+s_yiter]-y_value_0[s_j])/y_value_tick[m_y_cal_start+s_yiter])/delt_tick_tol 
           







