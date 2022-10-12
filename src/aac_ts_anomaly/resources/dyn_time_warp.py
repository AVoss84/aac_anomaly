
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wishart, multivariate_normal, bernoulli, multinomial
from scipy.sparse import csr_matrix
#from sklearn.model_selection import train_test_split
import os, pickle
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, normal, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand
from imp import reload
from copy import deepcopy
from math import cos, pi
#import seaborn as sns
import pandas as pd
import time
from scipy.spatial.distance import euclidean
import itertools
from itertools import chain, combinations


def cycle(N, nof_cycles = 1):
  return np.cos(2*pi*np.arange(0,N)*nof_cycles/N)

def simAR1(N,phi, sigma, burn=100):
  y = np.zeros((N+burn))
  for t in range(N+burn-1):
    y[t+1] = phi*y[t] + normal(scale = sigma, size=1)     
  return y[burn:]

np.random.seed(0)   # set seed

N = 200
omega = 1

y1 = omega*cycle(N, nof_cycles = 2) + simAR1(N, phi = 0.7, sigma = 0.6)

y2 = omega*cycle(N, nof_cycles = 2) + simAR1(N, phi = 0.7, sigma = 0.6)

y3 = omega*cycle(N , nof_cycles = 2) + simAR1(N , phi = 0.5, sigma = 1.4)

y4 = omega*cycle(N, nof_cycles = 2) 

# Plot trajectories:
#---------------------
plt.figure() ;pd.Series(y1).plot() ;pd.Series(y3).plot() ;plt.show()


class timewarp:
    """
    Dynamic time warping of two time series
    """
    def __init__(self, normalize=True, verbose = False, distance = ['manhattan','euclid']):                   
        self.verbose = verbose
        self.normalize = normalize
        self.distance = distance[0]

    def _normalize(self, y1, y2):
      """
      Normalize two time series so that distance/similarity values are comparable among each other
      http://luscinia.sourceforge.net/page26/page14/page14.html
      """
      y_mean_AB = (np.sum(y1) + np.sum(y2))/(len(y1)+len(y2))
      s_A = np.sum(np.square(y1 - y_mean_AB)) 
      s_B = np.sum(np.square(y2 - y_mean_AB)) 
      s_AB = np.sqrt((s_A + s_B)/(len(y1)+len(y2)-1))
      return (y1-y_mean_AB)/s_AB, (y2-y_mean_AB)/s_AB  

    def sim_determ_cycle(self, N, nof_cycles = 1):
      """Create determ. component for DGP"""
      return np.cos(2*pi*np.arange(0,N)*nof_cycles/N)

    def simAR1(self, N,phi, sigma, burn=100):
      """Create stochastic component for DGP"""
      y = np.zeros((N+burn))
      for t in range(N+burn-1):
        y[t+1] = phi*y[t] + normal(scale = sigma, size=1)     # Gaussian AR(1) process
      return y[burn:]

    # Adding local constraint on point matching (via window parameter):
    #-------------------------------------------------------------------
    def fit(self, y1, y2, window = 1):
        """
        Dynamic time warping of two time series with potentially different length
        """
        if self.normalize:  
            s,t = self._normalize(y1,y2)
        else:
            s,t = deepcopy(y1), deepcopy(y2) 
        if self.verbose:
          print('\nUsing distance type: {}'.format(self.distance))
          if self.normalize: print('Normalization applied.')
        n, m = len(s), len(t)
        w = np.max([window, abs(n-m)])
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0        
        for i in range(1, n+1):
            for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
                dtw_matrix[i, j] = 0
                if self.distance == 'manhattan' :
                  cost = abs(s[i-1] - t[j-1])                 # Manhattan distance
                else:  
                  cost = (s[i-1] - t[j-1])**2                 # Euclidean
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])   # recursions
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix, dtw_matrix[n,m]    
  
    def make_distmatrix(self, X : dict = None , series_list : list = None, col_selector : str = 'clm_cnt'): 
        """
        Create distance matrix for time series clustering
        Input:
        series_list: list of time series as in global namespace
        """        
        combis = list(itertools.combinations(series_list, r = 2))   # only duples
        matrix_index = list(itertools.combinations(range(0,len(series_list)), r = 2))   # only duples
        distmat = np.zeros((len(series_list),len(series_list)))
        distmat_pd = pd.DataFrame(distmat, index=series_list, columns=series_list)
        for index, co in zip(matrix_index, combis):
            if self.verbose:
              print('{} vs. {}'.format(co[0], co[1]))
            #x,y = eval(co[0]), eval(co[1])
            x = X[co[0]][col_selector].values
            y = X[co[1]][col_selector].values
            _, dist = eval("self.fit(x,y, 10)")
            distmat[index[0], index[1]] = dist
            distmat_pd.loc[co[0],co[1]] = dist
        return distmat_pd    

#dw = timewarp(normalize=True)
#res, dist = dw.fit(y1,y2, 5)

#dw.make_distmatrix(series_list = ['y1', 'y2', 'y3', 'y4'])


# Source: Wikipedia article
def dtw(s, t):
    """s,t: two time series of potentially different length"""
    n, m = len(s), len(t)
    #dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix = np.full((n+1, m+1), np.inf)
    #for i in range(n+1):
    #    for j in range(m+1):
    #        dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix, dtw_matrix[n,m] 

res, dist = dtw(y1,y2)

# Adding local constraint on point matching (via window parameter):
#-------------------------------------------------------------------
def dtw(s, t, window):
    """s,t: two time series of potentially different length"""
    n, m = len(s), len(t)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.full((n+1, m+1), np.inf)
    #dtw_matrix = np.zeros((n+1, m+1))
    #for i in range(n+1):
    #    for j in range(m+1):
    #        dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(s[i-1] - t[j-1])                 # Manhattan distance
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix, dtw_matrix[n,m]    

#start_time = time.time()
#res, dist = dtw(y1,y2, 10)
#print("--- %s seconds ---" % (time.time() - start_time))


# Distance matrix:
#-------------------
dd = np.zeros((4,4))
for z, i in enumerate(['y1', 'y2', 'y3', 'y4']):
 for r, j in enumerate(['y1', 'y2', 'y3', 'y4']):
    print(i, j)
    x = eval(i)
    y = eval(j)
    x,y = normalize(x,y)
    res, dist = eval("dtw(x,y,window=10)")
    dd[z,r] = dist

#stuff = ['y1', 'y2', 'y3', 'y4']

#stuff = list(get_all.keys())#[:3]

combis = list(itertools.combinations(stuff, r = 2))   # only duples
matrix_index = list(itertools.combinations(range(0,len(stuff)), r = 2))   # only duples

combis
matrix_index

# Distance matrix:
#------------------
distmat = np.zeros((len(stuff),len(stuff)))
#distmat = csr_matrix((len(stuff),len(stuff)), dtype=np.float32)#.toarray()
distmat_pd = pd.DataFrame(distmat, index=stuff, columns=stuff)

for index, co in zip(matrix_index, combis):
    print('{} vs. {}'.format(co[0], co[1]))
    x = get_all[co[0]]['clm_cnt'].values
    y = get_all[co[1]]['clm_cnt'].values
    #x,y = eval(co[0]), eval(co[1])
    res, dist = eval("timewarp(normalize=True).fit(x,y, 10)")
    distmat[index[0], index[1]] = dist
    distmat_pd.loc[co[0],co[1]] = dist

distmat_pd    
distmat

res, dist = dw.warp(y1,y2, 5)

csr_matrix((4, 4), dtype=np.float32)#.toarray()


#https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

for subset in all_subsets(stuff):
    print(subset)

def iter_pairs_no_lambda(indexes):
  def comp(x):
    return x[0] != x[1]
  return filter(comp, itertools.product(indexes, indexes))

res = iter_pairs_no_lambda(['y1', 'y2', 'y3','y4'])
