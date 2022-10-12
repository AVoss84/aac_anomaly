import numpy as np

# def agg_func(x):
#    """
#    Custom aggregation function for target variable
#    """ 
#    try:
#        mu = np.nansum(x)    
#        #mu = np.nanmean(x)   
#        #mu = np.nanmedian(x)
#        #maxi = np.nanmax(x)
#        #std = np.nanstd(x)
#        #z = (x-mu)/std
#        #return maxi/mu if mu != 0 else 0. 
#        #return np.nansum(z)  if std != 0 else np.nan
#        return mu
#    except RuntimeWarning:    
#         return np.nan
