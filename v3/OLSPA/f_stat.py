import numpy as np
import pandas as pd
from scipy.stats import f
from scipy import special
import statsmodels.api as sm
from scipy.optimize import fsolve
from statistics import mean, variance
from statsmodels.stats.outliers_influence import OLSInfluence 


##################################################################################
def create_data(n, mu, sigma, mde):
    """Creates a Pandas dataframe of simulated data from a normal distribution.

    Keyword arguments:
    n     -- sample size
    mu    -- mean
    sigma -- standard deviation
    mde   -- absolute effect size
    """
    a = list(np.random.normal(mu, sigma, n))
    b = list(np.random.rand(n))
    df = pd.DataFrame([a,b]).T
    df.columns = ['base', 'rand']
    df['Treated'] = np.where(df['rand'] <= 0.5, 1, 0)
    df['MDE']     = mde
    df['DV']      = np.where(df['Treated'] == 1, 
                             df['base'] + df['MDE'], 
                             df['base'])
    df = df[['DV','Treated', 'MDE']]
    return df

##################################################################################

def assess_power(df, candidate_n, rejection_region = 0.05, iterations = 100):
    

    p_vals = []
    for i in range(0, iterations, 1):
        data = df.sample(candidate_n) 
        X = data['Treated']
        X = sm.add_constant(X)
        Y = data['DV']
        model = sm.OLS(Y.astype(float), X.astype(float)).fit()
        p_vals.append(model.pvalues[1])
    
    results = []

    for j in p_vals:
        if j < rejection_region:
            results.append(1)
        else:
            results.append(0)
            
    return mean(results)

##################################################################################

def extract_r_delta(df):
    """Returns delta of r-squared with and without treatment variable.

    Keyword arguments:
    df  -- Pandas dataframe
    """
    X = df['Treated']
    X = sm.add_constant(X)
    Y = df['DV']
    model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()


    X['X'] = 1
    model2 = sm.OLS(Y.astype(float), X['X'].astype(float)).fit()

    return max(model1.rsquared, 0) - max(model2.rsquared, 0) 

##################################################################################

def get_f_stat_power(u, v, f2, sig_level = 0.05):
    """Returns power using the f-test.

    Keyword arguments:
    u  -- DF Numerator = (k-1) where k is the number of covariates + 1 for the intercept 
    v  -- DF Denominator = (n-k) where n is the total number of observations 
    f2 -- R2/(1−R2) where R2 is the coefficient of determination
    """
    return 1-special.ncfdtr(u, v, f2*(u+v+1), f.ppf(1-sig_level, u, v))

##################################################################################
def get_f_stat_n(u, f2, sig_level = 0.05):
    """Returns recommended sample size using the f-test.

    Keyword arguments:
    u  -- DF Numerator = (k-1) where k is the number of covariates + 1 for the intercept  
    f2 -- R2/(1−R2) where R2 is the coefficient of determination
    """
    def my_func(v):
        return 1-special.ncfdtr(u, v, f2*(u+v+1), f.ppf(1-sig_level, u, v))-0.8
    
    n = int(fsolve(my_func, 1000))
    if n == 1000:
        n = int(fsolve(my_func, 100))
    if n == 100:
        n = int(fsolve(my_func, 10))
    if n == 10:
        n = int(fsolve(my_func, 1))
    if n == 1:
        print("Failed to find a solution.")
        return None
    else:
        return n
    
##################################################################################
def acquire_sample_sizes_via_hueristic(df, mde, rejection_region = 0.05, desired_power = 0.8):

    X = df['Treated']
    X = sm.add_constant(X)
    Y = df['DV']
    model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()
    X['Constant'] = 1
    model2a = sm.OLS(model1.resid,                     X['Constant']).fit()
    model2b = sm.OLS(OLSInfluence(model1).resid_press, X['Constant']).fit()
    standard_resid_var = model2a.bse[0]*np.sqrt(X.shape[0])
    press_resid_var    = model2b.bse[0]*np.sqrt(X.shape[0])
    print("The s.e. of the regressed residuals is {}.".format(standard_resid_var))
    print("The s.e. of the regressed PRESS residuals is {}.".format(press_resid_var))
    effect_size_norm  = mde / standard_resid_var
    effect_size_press = mde / press_resid_var
    print("The estimated effect size via normal residuals is {}.".format(effect_size_norm))
    print("The estimated effect size via PRESS residuals is {}.".format(effect_size_press))

    recommended_n_norm = int(sm.stats.tt_ind_solve_power(effect_size = effect_size_norm, 
                             alpha = rejection_region, power = desired_power, 
                             alternative = 'larger'))
    recommended_n_press = int(sm.stats.tt_ind_solve_power(effect_size = effect_size_press, 
                              alpha = rejection_region, power = desired_power, 
                              alternative = 'larger'))
    return recommended_n_norm, recommended_n_press