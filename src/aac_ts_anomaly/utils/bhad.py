from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import CategoricalDtype
import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.stats import wishart, multivariate_normal, bernoulli, multinomial
import os, sys, warnings, functools, math, time
from math import floor, ceil
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import wraps


class BHAD(BaseEstimator, OutlierMixin):
    """
    Bayesian Histogram-based Anomaly Detector (BHAD), see [1,2] for details. 

    Parameters
    ----------
    contamination : float, optional (default=0.1)
        The amount (fraction) of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    Attributes
    ----------
    threshold : float
        The outlier score threshold calculated based on the score distribution 
        and the provided contamination argument.
    value_counts : dictionary of str keys and pandas.Series values
        A dictionary with column names as the keys and pandas.Series as the values. 
        Each pandas.Series has the value count of each observation in the respective 
        column. The observations (str) are the indices in the pandas.Series object 
        and the counts (int) are the respective values.

    Methods
    -------
    decision_function(self, X[, y])
        Outlier scores of X based on algorithm centered 
        around threshold value. 

    fit(self, X[, y])
        Fit the model. 

    fit_predict(self, X[, y])
        Performs fit on X and returns outlier labels for X. This function is 
        sklearn compatible.

    predict(self, X)
        Predict if a particular sample is an outlier (-1 label) or not (1 label). 

    score_samples(self, X)
        Score of the samples as the outlier score calculated by summing the counts 
        of each feature level in the dataset.
        Although this function is consistent with the sklearn OutlierMixin class, 
        it can not be used with sklearn pipelines (as is the case with other sk-learn 
        outlier detector classes).

    Reference:
    ------------
    [1] Vosseler, A. (2022): Unsupervised insurance fraud prediction based on anomaly detector ensembles, Risks, 10 (132)
    [2] Vosseler, A. (2021): BHAD: Fast unsupervised anomaly detection using Bayesian histograms, Technical Report
    """

    def __init__(self, contamination : float = 0.01, alpha : float = 1/2, exclude_col : list = [], append_score : bool = False, verbose : bool = True):
        
        self.contamination = contamination                   # outlier proportion in the dataset
        self.alpha = alpha                              # uniform Dirichlet prior concentration parameter used for each feature
        self.verbose = verbose
        self.append_score = append_score
        self.exclude_col = exclude_col               # list with column names in X of columns to exclude for computation of the score
        super(BHAD, self).__init__()

    def __del__(self):
        class_name = self.__class__.__name__
    
    #@timer
    def _fast_bhad(self, X : pd.DataFrame)-> pd.DataFrame:
      """
      Input:
      X:            design matrix as pandas df with all features (must all be categorical, 
                    since one-hot enc. will be applied! Otherwise run discretize() first.)
      append_score: Should anomaly score be appended to X?
      return: scores
      """  
      assert isinstance(X, pd.DataFrame)
      selected_col = X.columns[~X.columns.isin(self.exclude_col)] 
      if len(self.exclude_col)>0:
         print("Features",self.exclude_col, 'excluded.')  
        
      df = deepcopy(X[selected_col]) 
      self.df_shape = df.shape  
      self.columns = df.select_dtypes(include='object').columns.tolist()  # use only categorical (including discretized numerical)
      if len(self.columns)!= self.df_shape[1] : warnings.warn('Not all features in X are categorical!!')
      self.df = df
      unique_categories_ = [df[var].unique().tolist() + ['infrequent'] for var in df.columns]
      #self.enc = OneHotEncoder(handle_unknown='infrequent_if_exist', dtype = int, categories = unique_categories_)
      self.enc = onehot_encoder(prefix_sep='__')   # current performance bottleneck
      self.df_one = self.enc.fit_transform(df)#.toarray()   # apply one-hot encoder to categorical -> sparse dummy matrix
      assert all(np.sum(self.df_one, axis=1) == df.shape[1]), 'Row sums must be equal to number of features!!'
      if self.verbose : print("Matrix dimension after one-hot encoding:", self.df_one.shape)  
           
      self.columns_onehot_ = self.enc.get_feature_names()

      self.alphas = np.array([self.alpha]*self.df_one.shape[1])        # Dirichlet concentration parameters; aka pseudo counts
      self.freq = self.df_one.sum(axis=0)                                 # suff. statistics of multinomial likelihood
      self.log_pred = np.log((self.alphas + self.freq)/np.sum(self.alphas + self.freq))  # log posterior predictive probabilities for single trial / multinoulli

      # Duplicate list of marg. freq. in an array for elementwise multiplication  
      # i.e. Repeat counts for each obs. i =1...n
      #---------------------------------------------------------------------------   
      # Keep frequencies for explanation later         
      a = np.tile(self.freq, (self.df_shape[0], 1))     # Repeat counts for each obs. i =1...n
      a_bayes = np.tile(self.log_pred, (self.df_shape[0], 1))

      # Keep only nonzero matrix entries 
      # (via one-hot encoding matrix), i.e. freq. for respective entries
      # Assign each obs. the overall category count
      #-------------------------------------------------------------------
      self.f_mat = self.df_one * np.array(a)                    # keep only nonzero matrix entries
      f_mat_bayes = self.df_one * np.array(a_bayes)

      # Calculate outlier score for each row (observation), see equation (5) in paper.
      out = pd.Series(np.apply_along_axis(np.sum, 1, f_mat_bayes), index=df.index)    
      if self.append_score:  
         out = pd.concat([df, pd.DataFrame(out, columns = ['outlier_score'])], axis=1)
      return out    
    

    def fit(self, X : pd.DataFrame, y=None):
        """
        Apply the BHAD and calculate the outlier threshold value.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or castable to 
            str (e.g. catagorical).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BHAD object
        """
        if self.verbose : print("\nConstruct Bayesian Histogram-based Anomaly Detector (BHAD)")
        self.scores = self._fast_bhad(X)
    
        if self.append_score:  
            self.threshold = np.nanpercentile(self.scores['outlier_score'].tolist(), q=100*self.contamination)
        else: 
            self.threshold = np.nanpercentile(self.scores.tolist(), q=100*self.contamination)
        if self.verbose : print("BHAD completed.")

        self.scores_ = self.scores          
        self.threshold_ = self.threshold
        self.freq_ = self.freq
        self.f_mat_ = self.f_mat
        self.df_one_ = self.df_one 
        self.X_ = X
        self.df_ = self.df
        self.xindex_fitted_ = X.index
        self.enc_ = self.enc
        return self

    
    def score_samples(self, X : pd.DataFrame)-> pd.DataFrame:
        """
        Outlier score calculated by summing the counts 
        of each feature level in the dataset.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold 
            value.
        """
        df = deepcopy(X)
        self.df_one = self.enc_.transform(df)#.toarray()   # apply fitted one-hot encoder to categorical -> sparse dummy matrix
        assert all(np.sum(self.df_one, axis=1) == df.shape[1]), 'Row sums must be equal to number of features!!'

        self.freq_updated_ = self.freq_ + self.df_one.sum(axis=0)      # update suff. stat with abs. freq. of new data levels
        #freq_updated = np.log(np.exp(self.freq) + self.df_one + alpha)    # multinomial-dirichlet
        self.log_pred = np.log((self.alphas + self.freq_updated_)/np.sum(self.alphas + self.freq_updated_))   # log posterior predictive probabilities for single trial / multinoulli
        self.f_mat = self.freq_updated_ * self.df_one           # get level specific counts for X, e.g. test set
        f_mat_bayes = self.log_pred * self.df_one  
        self.scores = pd.Series(np.apply_along_axis(np.sum, 1, f_mat_bayes), index=X.index) 

        # If you already have it from fit then just output it:
        if hasattr(self, 'X_') and (len(self.xindex_fitted_) == X.shape[0]):
            self.f_mat = deepcopy(self.f_mat_)
            return self.scores_
        else:    
            return self.scores
    
    
    def decision_function(self, X : pd.DataFrame) -> np.array:
        """
        Outlier score centered around the threshold value. Outliers are scored 
        negatively (<= 0) and inliers are scored positively (> 0).

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold 
            value.
        """
        score = self.score_samples(X).to_numpy()
        # center scores; divide into outlier and inlier (-/+)
        return score - self.threshold_
    
    
    def predict(self, X : pd.DataFrame) -> np.array:
        """
        Returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : array, shape (n_samples,)
            The outlier labels of the input samples.
            -1 means an outlier, 1 means an inlier.
        """
        self.anomaly_scores = self.decision_function(X)            # get centered anomaly scores
        outliers = np.asarray(-1*(self.anomaly_scores <= 0).astype(int))
        inliers = np.asarray((self.anomaly_scores > 0).astype(int))
        return outliers + inliers



# timer decorator for any function func:
def timer(func):
    """Print the runtime of the decorated function"""
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


# Mimics R's paste() function for two lists:
#---------------------------------------------
def reduce_concat(x, sep : str = ""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep : str = " ", collapse : str = None) -> list:
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)


def jitter(M: int, noise_scale: float = 10**5., seed : int = None)-> np.array:

  """ Generates jitter that can be added to any float, e.g.
      helps when used with pd.qcut to produce unique class edges
      M: number of random draws, i.e. size
  """  
  if seed is not None:
     np.random.seed(seed)
  return np.random.random(M)/noise_scale


#@timer
class discretize(BaseEstimator, TransformerMixin):
    """
    Discretize continous features by binning. Will be used as input for Bayesian histogram anomaly detector (BHAD)
    
    Input:
    ------
    columns: list of feature names
    nbins: number of bins to discretize numeric features into
    lower: optional lower value for the first bin, very often 0, e.g. amounts
    k: number of standard deviations to be used for the intervals, see k*np.std(v)  
    round_intervals: number of digits to round the intervals
    eps: minimum value of variance of a numeric features (check for 'zero-variance features') 
    make_labels: assign integer labels to bins instead of technical intervals
    """
    def __init__(self, columns : list = [], nbins : int = None, lower : float = None, k : int = 1, 
                 round_intervals : int = 5, eps : float = .001, 
                 make_labels : bool = False, 
                 verbose : bool = True, prior_gamma : float = 0.9, prior_max_M : int = 50,  # Bayesian AVF (estimate number of bins M)
                 **kwargs):
        
        self.columns = columns 
        self.round_intervals = round_intervals
        self.nof_bins = nbins #-1    # there will be +/- Inf endpoints added; correct for that
        self.nbins, self.xindex_fitted, self.df_orig = None, None, None
        self.counts_binned, self.save_binnings = dict(), dict()
        self.lower = lower 
        self.verbose = verbose
        self.k = k
        self.prior_gamma = prior_gamma
        self.prior_max_M = prior_max_M
        self.eps = eps                           # threshold for close-to-zero-variance
        self.make_labels = make_labels
        
        if self.lower and (self.lower != 0):
            if self.verbose : warnings.warn("'\nNote: lower != 0 not supported currently, will be set to None!'")
            self.lower = None 
    

    def __del__(self):
        class_name = self.__class__.__name__
        #print(class_name, "destroyed")

    def fit(self, X : pd.DataFrame, y=None):
        
            assert isinstance(X, pd.DataFrame), 'Input X must be pandas dataframe!'
            self.nbins = self.nof_bins    # initialize (might be changed in case of low variance features)
            df_new = deepcopy(X) 
            self.cat_columns = df_new.select_dtypes(include='object').columns.tolist()  # categorical (for later reference in postproc.)
            if not self.columns:
                self.columns = df_new.select_dtypes(include=[np.number]).columns.tolist()    # numeric features only

            if self.verbose: print(f"Used {len(self.columns)} numeric feature(s) and {len(self.cat_columns)} categorical feature(s).")       
            df_new[self.columns] = df_new[self.columns].astype(float)        
            ptive_inf = float ('inf') ; ntive_inf = float('-inf')
            self.df_orig = deepcopy(df_new[self.columns + self.cat_columns])   # train data with non-discretized values for numeric features for model explainer

            for col in self.columns:
                
                    v = df_new[col].values
                    
                    # Determine optimal number of bins per feature:
                    #-----------------------------------------------
                    if self.nof_bins is None:
                        if self.verbose : print("Determining optimal number of bins via Bayesian MAP estimate")
                        #self.nbins = 1 + ceil(np.log2(len(v)))    # use Sturge's rule for number of bins per variable
                        #self.nbins = utils.freedman_diaconis(v)    # use FD rule
                        #print(f'FD rule: {utils.freedman_diaconis(v)}')
                        #lpr = {m:utils.log_post_nbins(m, v) for m in range(1,80, 1)}   # own Bayesian Block method
                        lpr = {m:(log_post_nbins(m, v) + np.log(geometric_prior(m, gamma = self.prior_gamma, max_M = self.prior_max_M))) for m in range(1,self.prior_max_M, 1)}
                        self.nbins = max(lpr, key=lpr.get)    # compute K_MAP for each feature, see paper
                        if self.verbose: print('Feature {} using {} bins'.format(col, self.nbins))
                    
                    # Add some low variance white noise 
                    # to make bins unique (thus more robust):
                    #------------------------------------------
                    if (np.nanvar(v) < self.eps):                # check for close to zero-variance feature .np.nanstd(v)
                        self.nbins = len(set(v))
                        v += jitter(M = len(v), noise_scale = 10**5.)
                        if self.verbose: print("{} has close to zero variance - jitter applied! Overwriting nbins to {}.".format(col, self.nbins))

                    if (self.lower == 0) & any(v < 0):
                        if self.verbose: print("Replacing negative values for", col)
                        v = np.where(v<0, 0, v)
                    # Define knots for the binning (equally spaced) -> histogram
                    if self.lower is None:
                      bounds = np.linspace(min(v)-self.k*np.std(v), max(v)+self.k*np.std(v), num = self.nbins+1)  
                      bs, labels = [], ['0']    # add -Inf as lower bound
                    else:
                      bounds = np.linspace(self.lower, max(v)+self.k*np.std(v), num = self.nbins+1)
                      bs, labels = [],[]

                    bs = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]    

                    # Add +Inf as upper bound    
                    #--------------------------
                    bs.insert(len(bs),(bounds[len(bounds)-1], ptive_inf))   
                    
                    # Add -Inf as lower bound    
                    #--------------------------
                    if self.lower is None: bs[0] = (ntive_inf, bs[0][1]) 
                      
                    # Make left closed [..) interval to be save in cases like: [0,..)
                    #--------------------------------------------------------------------
                    my_bins = pd.IntervalIndex.from_tuples(bs, closed='left')   
                    self.save_binnings[col] = my_bins
                    assert (self.nbins + 1) == len(set(my_bins)), 'Your created bins in '+str(col)+' are not unique my_bins!'
                    x = pd.cut(v, bins = my_bins, duplicates = "drop", include_lowest = True)
                    if self.make_labels : x.categories = [str(i) for i in np.arange(1,len(bs)+1)]         # set integer labels
                    df_new[col] = x
                    df_new[col] = df_new[col].astype(object)
                    self.counts_binned[col] = df_new[col].value_counts()
                    if self.nof_bins is not None: self.nbins = self.nof_bins             # resetting nbins, in case of zero variance features...

            # Tag as fitted for sklearn compatibility: 
            # https://scikit-learn.org/stable/developers/develop.html#estimated-attributes
            self.X_ = deepcopy(df_new)
            self.columns_ = self.columns
            self.nbins_ = self.nbins 
            self.xindex_fitted_ = df_new.index
            self.save_binnings_ = self.save_binnings 
            self.counts_binned_ = self.counts_binned 
            self.lower_ = self.lower 
            self.k_ = self.k 
            self.eps_ = self.eps 
            self.make_labels_ = self.make_labels 
            self.df_orig_ = deepcopy(self.df_orig)
            if self.verbose and (self.nof_bins is not None): print("Binned continous features into", self.nbins,"bins.")
            return self
    
    
    def transform(self, X : pd.DataFrame, y=None):
        
        df_new = deepcopy(X)
        self.cat_columns = df_new.select_dtypes(include='object').columns.tolist()  # categorical (for later reference in postproc.)
        x_columns = df_new.select_dtypes(include=[np.number]).columns.tolist()    # numeric features only

        # Check if columns in X and X_train are compatible:
        for col in x_columns : assert col in self.columns_, 'Column {} not among loaded X_train columns!'.format(col)
        df_new[self.columns_] = df_new[self.columns_].astype(float)   
        # Update & Keep for model explainer
        self.df_orig = deepcopy(df_new[self.columns_ + self.cat_columns])  
        # if you already have it from fit then just output it
        if hasattr(self, 'X_') and (len(self.xindex_fitted_) == X.shape[0]):
            return self.X_

        # Encode/map new values to discrete training buckets/bins: 
        for ind, row in df_new.iterrows(): 
            row_values = []
            try:
                for c in self.columns_:
                    bin_c = self.save_binnings_[c]
                    if ~np.isnan(row[c]):
                        row_values.append(list(bin_c[bin_c.contains(row[c])])[0])
                    else:
                        row_values.append(np.nan)    
                df_new.loc[ind, self.columns_] = row_values
            except Exception as ex:
                print(ex)        
        return df_new  


def log_post_nbins(M : int, y : np.array):
      """
      Log posterior of number of bins M
      using conjugate Jeffreys' prior for the bin probabilities 
      and a flat improper prior for the number of bins. This is therefore equivalent to the marginal
      log-likelihood of the number of bins  
      """
      N = len(y)
      counts, bin_edges = np.histogram(y, bins = np.linspace(min(y), max(y), M+1))   # evenly spaced bins
      post_M = N*np.log(M) + loggamma(M/2) -M*loggamma(1/2) -loggamma(N + M/2) + np.sum(loggamma(counts + 1/2)) 
      return post_M


def geometric_prior(M, gamma : float = 0.7, max_M : int = 100):
  """
  Geometric (power series) prior p.m.f. of M
  """
  #gamma = (gamma < 1)*gamma    # assuming |gamma| < 1 for convergence of the series
  gamma = gamma if 0 < gamma < 1 else 0    # indicator function according to uniform prior 
  P0 = (1-gamma)/(1-gamma**(max_M))
  return P0*(gamma**M)



class mvt2mixture:
    
    def __init__(self, thetas : dict = {'mean1' : None, 'mean2' : None, \
                               'Sigma1' : None, 'Sigma2' : None, \
                               'nu1': None, 'nu2': None}, seed : int = None, gaussian : bool = False, **figure_param):
        """
        Multivariate 2-component Student-t mixture random generator. 
        Direct random sampling via using the Student-t representation as a continous scale mixture distr.   
        -------
        Input:
        -------
        thetas: Component-wise parameters; note that Sigma1,2 are the scale matrices of the 
                Wishart priors of the precision matrices of the Student-t's.
        gaussian: boolean, generate from Gaussian mixture if True, otherwise from Student-t    
        seed: set seed for rng.
        """
        self.thetas = thetas ; self.seed = seed ; self.gaussian = gaussian; self.para = figure_param
        if self.seed is not None:
            np.random.seed(seed)
    
    def draw(self, n_samples = 100, k = 2, p = .5): 
        """
        Random number generator:
        Input:
        -------
        n_samples: Number of realizations to generate
        k:         Number of features (Dimension of the t-distr.)
        p:         Success probability Bernoulli(p) p.m.f. 
        """
        self.n_samples = n_samples ; self.k = k; self.p = p ; m = 2                # number of mixture components
        assert (len(self.thetas['mean1']) == k) & (self.thetas['Sigma1'].shape[0] == k), 'Number of dimensions does not match k!'

        if self.gaussian:
            cov1, cov2 = self.thetas['Sigma1'], self.thetas['Sigma2']  
        else:    
            cov1 = wishart.rvs(df = self.thetas['nu1'], scale = self.thetas['Sigma1'], size=1)
            cov2 = wishart.rvs(df = self.thetas['nu2'], scale = self.thetas['Sigma2'], size=1)

        self.var1 = self.thetas['nu1']/(self.thetas['nu1']-2)*cov1       # variance covariance matrix of first Student-t component
        self.var2 = self.thetas['nu2']/(self.thetas['nu2']-2)*cov2
        self.phi_is = bernoulli.rvs(p = self.p, size = self.n_samples)          # m=2
        Phi = np.tile(self.phi_is, self.k).reshape(self.k,self.n_samples).T    # repeat phi vector to match with random matrix
        rn1 = np.random.multivariate_normal(self.thetas['mean1'], cov1, self.n_samples)
        rn2 = np.random.multivariate_normal(self.thetas['mean2'], cov2, self.n_samples)
        self.sum1 = np.multiply(Phi, rn1)
        self.sum2 = np.multiply(1-Phi, rn2)
        self.x_draws = np.add(self.sum1,self.sum2)
        return self.phi_is, pd.DataFrame(self.x_draws,columns = paste(['var']*self.k, list(np.arange(self.k)), sep=""))


    def show2D(self, save_plot=False, legend_on = True, **kwargs):
        """
        Make scatter plot for first two dimensions of the random draws
        """
        x_comp1,y_comp1 = self.sum1[:,0], self.sum1[:,1]
        x_comp2,y_comp2 = self.sum2[:,0], self.sum2[:,1]
        fig = plt.figure(**self.para) ; 
        la = plt.scatter(x_comp1, y_comp1, c="blue", **kwargs)
        lb = plt.scatter(x_comp2, y_comp2, c="orange", **kwargs)
        lc = plt.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
                         [self.thetas['mean1'][1],self.thetas['mean2'][1]], c="black", s=6**2, alpha=.5)
        #plt.title("Draws from 2-component \nmultivariate Student-t mixture \n(first two dimensions shown)")
        plt.xlabel(r'$x_{1}$') ; plt.ylabel(r'$x_{2}$')
        if legend_on:
            plt.legend((la, lb), ('Outlier', 'Inlier'),
                            scatterpoints=1, loc='lower right', ncol=3, fontsize=8)
        plt.show() 
        if save_plot:
            fig.savefig('mixturePlot2D.jpg')
            print("Saved to:", os.getcwd())


    def show3D(self, save_plot=False, legend_on = True, **kwargs):
        """
        Make scatter plot for first three dimensions of the random draws
        """
        fig = plt.figure(**self.para) ; ax = Axes3D(fig)
        x_comp1,y_comp1, z_comp1 = self.sum1[:,0], self.sum1[:,1], self.sum1[:,2]
        x_comp2,y_comp2, z_comp2 = self.sum2[:,0], self.sum2[:,1], self.sum2[:,2]
        la = ax.scatter(x_comp1, y_comp1, z_comp1, c="blue", **kwargs) 
        lb = ax.scatter(x_comp2, y_comp2, z_comp2, c="orange", **kwargs)  
        lc = ax.scatter([self.thetas['mean1'][0], self.thetas['mean2'][0]], 
                     [self.thetas['mean1'][1],self.thetas['mean2'][1]], 
                     [self.thetas['mean1'][2],self.thetas['mean2'][2]], c="black", s=6**2, alpha=.2)

        #plt.title("Draws from 2-component \nmultivariate mixture \n(first three dimensions shown)")
        ax.set_xlabel(r'$x_{1}$') ; ax.set_ylabel(r'$x_{2}$') ;ax.set_zlabel(r'$x_{3}$')
        if legend_on:
            ax.legend((la, lb), ('Outlier', 'Inlier'),
                        scatterpoints=1, loc='lower left', ncol=3, fontsize=8)    
        plt.show()
        if save_plot:
            fig.savefig('mixturePlot3D.jpg')
            print("Saved to:", os.getcwd())



@timer
class onehot_encoder(TransformerMixin, BaseEstimator):

    def __init__(self, exclude_columns=[], prefix_sep = '_', oos_token = 'OTHERS', verbose = True, **kwargs):
        """
        One-hot encoder that handles out-of-sample levels of categorical variables
        Args:
            oos_token (str, optional): [description]. Defaults to 'OTHERS'.
        """
        self.oos_token_ = oos_token
        self.kwargs = kwargs
        self.exclude_col = exclude_columns
        self.prefix_sep_ = prefix_sep
        self.verbose = verbose
        self.unique_categories_, self.value2name_ = dict(), dict()
        if self.verbose : print("One-hot encoding of categorical features")

    def fit(self, X):

        self.selected_col = X.columns[~X.columns.isin(self.exclude_col)] 
        if len(self.exclude_col)>0:
            print("Features",self.exclude_col, 'excluded.')  
        df = deepcopy(X[self.selected_col])
        for z, var in enumerate(df.columns):
            self.unique_categories_[var] = df[var].unique().tolist()
            # Add 'Unknown/Others' bucket to levels for unseen levels:
            df[var] = df[var].astype(CategoricalDtype(self.unique_categories_[var] + [self.oos_token_]))    # add unknown catgory for out of sample levels
            one = pd.get_dummies(df[var], prefix_sep = self.prefix_sep_, prefix=var, **self.kwargs)
            if z>0:
               dummy = pd.concat([dummy, one], axis=1, sort=False)
            else:
               dummy = deepcopy(one)

            # Leave out the 'OTHERS'/oos_token_ buckets here for consistency:
            self.value2name_[var] = {level_orig:dummy_name for level_orig, dummy_name in zip(self.unique_categories_[var], list(one.columns)[:-1])}

        self.dummyX_ = dummy #csr_matrix(dummy)           
        self.columns_ = list(self.dummyX_.columns) # all final column names in sparse dummy matrix
        self.names2index_ = {dummy_names:z for z, dummy_names in enumerate(self.columns_)} 
        self.X_ = df
        return self    

    def transform(self, X)-> np.array:
        
        check_is_fitted(self)        # Check if fit had been called
        self.selected_col = X.columns[~X.columns.isin(self.exclude_col)]

        # If you already have it from fit then just output it
        if hasattr(self, 'X_') and self.X_.equals(X):
            return self.dummyX_
 
        df = deepcopy(X[self.selected_col])
        ohm = np.zeros((df.shape[0],len(self.columns_)))
        for r, my_tuple in enumerate(df.itertuples(index=False)): 
            my_index = []
            for z, col in enumerate(df.columns):
                raw_level_list = list(self.value2name_[col].keys())
                mask = my_tuple[z] == np.array(raw_level_list)
                if any(mask): 
                    index = np.where(mask)[0][0]
                    dummy_name = self.value2name_[col][raw_level_list[index]]
                else:
                    dummy_name = col + self.prefix_sep_ + self.oos_token_
                my_index.append(self.names2index_[dummy_name])
            targets = np.array(my_index).reshape(-1)
            ohm[r,targets] = 1    
        return ohm #pd.DataFrame(ohm,columns=self.columns_)
        

    def get_feature_names(self, input_features : list = None)-> np.array: 

        """
        Get feature names as used in one-hot encoder, 
        i.e. after binning/disretizing

        Returns:
            [numpy array]: feature names as used in discretizer, e.g. intervals
        """

        check_is_fitted(self)
        if input_features is None:
            input_features = self.selected_col
        self.dummy_names = []; self.dummy_names_index = {}; self.dummy_names_by_feat = {}
        for col_name in input_features:
            mask = [col_name in col for col in self.columns_]
            if any(mask):
                index = np.where(np.array(mask))[0]
                self.dummy_names_index[col_name] = index
                full_dummy_names_feat = np.array(self.columns_)[index].tolist()
                # Remove separator from pandas get_dummy to retrieve original bin names from discretize fct.:
                # Note: self.prefix_sep_ = '__' must be kept here!
                dummy_name_2_orig_val = [item.split('__')[1] for item in full_dummy_names_feat] #{item: item.split('__')[1] for item in full_dummy_names_feat}
                self.dummy_names_by_feat[col_name] = dummy_name_2_orig_val
                self.dummy_names += self.dummy_names_by_feat[col_name]
        return np.array(self.dummy_names, dtype=object)  


