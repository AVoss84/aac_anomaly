from sklearn.base import BaseEstimator, OutlierMixin
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from tqdm import tqdm


class BayesOCPD(BaseEstimator, OutlierMixin):
    """
    Bayesian online change point detection, see Adams & MacKay 2007.
    """
    def __init__(self, model, hazard, mini_run_length : int = 10, verbose : bool = True):
        self.verbose = verbose
        self.model = model
        self.hazard = hazard
        self.mini_run_length = mini_run_length
        if self.verbose : print("*** Bayesian online change point detection ***")

    def fit(self, X, y=None):    
        """
        Return run length posterior using Algorithm 1
        """
        #data        = X.to_numpy().copy()
        #ts_index    = X.index
        data        = X.copy()
        T           = len(data)
        log_R       = -np.inf * np.ones((T+1, T+1))    # log posterior values of run length at time t
        log_R[0, 0] = 0              # log 0 == 1
        pmean       = np.empty(T)    # Model's predictive mean.
        pvar        = np.empty(T)    # Model's predictive variance. 
        log_message = np.array([0])  # log 0 == 1
        log_H       = np.log(self.hazard)
        log_1mH     = np.log(1 - self.hazard)
        cps_MAP     = np.empty(T) 

        for t in tqdm(range(1, T+1)):       # forward filtering 
            # Observe new datum.
            x = data[t-1]

            # Make model predictions. ('step 9'). Calculate first two moments
            pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * self.model.mean_params[:t])   # up to t-1
            pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * self.model.var_params[:t])
            
            # Evaluate predictive posterior probabilities.
            log_pis = self.model.log_pred_prob(t, x)

            # Calculate growth probabilities ('No break')
            log_growth_probs = log_pis + log_message + log_1mH

            # Calculate changepoint/Break probabilities.
            log_cp_prob = logsumexp(log_pis + log_message + log_H)

            # Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)    # append: pr(r_t == 0), pr(r_t > 0)'s

            # Determine run length posterior mass distribution.
            log_R[t, :t+1]  = new_log_joint
            log_R[t, :t+1] -= logsumexp(new_log_joint)            # normalize entries by dividing by column sum

            # Update sufficient statistics.
            # here: mean_params & prec_params
            self.model.update_params(t, x)

            # Pass message.
            log_message = new_log_joint
            # Select break points, via MAP state estimate        
            cps_MAP[t-1] = np.argmax(log_R[t-1,:t])
        
        self.pmean = pmean
        self.pvar = pvar
        self.R = np.exp(log_R)
        self.log_R = log_R
        self.cps_MAP = cps_MAP
        return self
    
    # Future: change this to be able run for online oob samples
    def predict(self, X):
        lab = np.ones(len(X))
        lab[self.score(X)] = -1
        return lab

    def score(self, X):    
        self.cps_est = np.where((self.cps_MAP[1:] - self.cps_MAP[:-1]) < -self.mini_run_length)[0]    # MAP estimates of change points
        return self.cps_est
        
        
#-----------------------------------------------------------------------------

class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """
        Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0               # mu_0
        self.var0  = var0                # sigma2_0
        self.varx  = varx                # sigma2
        self.mean_params = np.array([mean0])      # mu_n   ; initialize posterior param. with prior hyperparameters -> used for posterior predictive distr. \pi_t (step 3)
        self.prec_params = np.array([1/var0])     # lambda_n
    
    def log_pred_prob(self, t, x):
        """
        Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])        #  use current variance parameter attribute
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """
        Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)         # lambda_n (27)
        self.prec_params = np.append([1/self.var0], new_prec_params)   # recursively update precision parameters

        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)    # recursively update mean parameters

    @property
    def var_params(self):
        """Helper function for computing the posterior variance sigma2_n
        """
        return 1./self.prec_params + self.varx  # can be accessed as attribute var_params, rather than a function call var_params()

# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """
    Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)    # draw a new mean parameter with prob. cp_prob
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, cps, cps_estimated, R, pmean, pvar, plot_est_cps = True):
    
    fig, axes = plt.subplots(2, 1, figsize=(20,10))
    ax1, ax2 = axes
    # Plot raw data
    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    # Plot predictions mean and confidence bounds based on Gaussian posterior predictive distr.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)
    
    if cps is not None: 
        for cp in cps:
            ax1.axvline(cp, c='red', ls='--')
            ax2.axvline(cp, c='red', ls='--')

    if plot_est_cps: 
      for cp in cps_estimated:
          ax1.axvline(cp, c='green', ls='dotted')
          ax2.axvline(cp, c='green', ls='dotted')

    plt.tight_layout()
    plt.show()

   
#T      = 1000   # Number of observations.
#hazard = .01  # Constant prior on changepoint probability.
#mean0  = 0      # The prior mean on the mean parameter.
#var0   = 2      # The prior variance for mean parameter.
#varx   = 1      # The known variance of the data.
#data, cps      = generate_data(varx, mean0, var0, T, hazard)
#model          = GaussianUnknownMean(mean0, var0, varx)
#R, pmean, pvar, cps_MAP, cps_est = bocd(data, model, hazard, mini_run_length = T*.05)   





        
        
