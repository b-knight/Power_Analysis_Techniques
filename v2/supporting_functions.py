import os
import time
import random 
import numpy as np
import pandas as pd
from statistics import mean 
import statsmodels.api as sm
from datetime import datetime   
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression

################################################################################################
def create_sim_data_base(absolute_treatment_effect, num_orders, num_retailers, 
                         retailer_loc, retailer_scale, noise_loc, noise_scale):
    """Creates simulated data. Called by create_sim_data().

    Keyword arguments:
    absolute_treatment_effect -- the absolute value of the difference across groups A/B 
    num_orders -- number of "orders" (observations) to be created
    num_retailers -- number of distinct retailers to create  
    retailer_loc -- mean of the Numpy random normal distribution used by retailers
    retailer_scale -- standard deviation of the Numpy random normal distribution used by retailers
    noise_loc -- mean of the Numpy random normal distribution used to create random noise
    noise_scale -- standard deviation of the Numpy random normal distribution used to create random noise
    """
    # Create retailers, their order amount scalars, and the probability space.
    retailer_ids = list(range(1, num_retailers + 1,1))
    retailer_scalars = list(np.random.normal(loc = retailer_loc, scale = retailer_scale, 
                                             size = num_retailers))
    retailers = pd.DataFrame(list(zip(retailer_ids, retailer_scalars)), 
                             columns =['Retailer_ID', 'Mean_Retailer_Amount']) 
    delimiters = list(np.linspace(0.0, 1.0, num = num_retailers + 1))
    retailers['a'] = delimiters[0:len(delimiters)-1]
    retailers['b'] = delimiters[1:len(delimiters)]

    # Create initial orders
    order_ids = pd.DataFrame(list(range(1, num_orders + 1,1)))
    order_ids.columns = ['Order_Id']

    # Determine treatment assignment
    treatment_probs = []
    for i in list(range(0, len(order_ids), 1)):
        treatment_probs.append(np.random.uniform())
    order_ids['Prob']    = treatment_probs
    order_ids['Treated'] = np.where(order_ids['Prob'] <= 0.5, 1, 0)
    order_ids['ABS_MDE'] = np.where(order_ids['Prob'] <= 0.5, absolute_treatment_effect, 0.0)

    # Assign retailers
    a  = order_ids['Prob']
    bh = retailers.b.values
    bl = retailers.a.values
    i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))
    order_ids = pd.DataFrame(np.column_stack([order_ids.values[i], order_ids.values[j]]))
    order_ids.columns = ['Order_ID', 'Prob', 'Treated', 'ABS_MDE', 'Retailer_ID', 'X', 'Y', 'Z']
    order_ids = order_ids[['Order_ID', 'Treated', 'ABS_MDE', 'Retailer_ID']]
    order_ids = pd.merge(order_ids, retailers, on = 'Retailer_ID')

    # Add random noise 
    order_ids['Noise'] = list(np.random.normal(loc = noise_loc, scale = noise_scale, size = len(order_ids)))
    order_ids['Initial_Order_Amount'] = order_ids['Mean_Retailer_Amount'] + \
                                        order_ids['Noise'] + order_ids['ABS_MDE']

    # clean data
    order_ids = order_ids[['Order_ID', 'Initial_Order_Amount', 'Treated', 'Mean_Retailer_Amount']]
    return order_ids


################################################################################################
def create_sim_data(absolute_treatment_effect, num_orders, num_retailers, retailer_loc,             
                    retailer_scale, noise_loc, noise_scale):

    if not os.path.exists('data'):
        os.makedirs('data') 

    labels = ['absolute_treatment_effect', 'num_orders', 'num_retailers', 
              'retailer_loc', 'retailer_scale', 'noise_loc', 'noise_scale']
    values = [absolute_treatment_effect, num_orders, num_retailers, retailer_loc,             
              retailer_scale, noise_loc, noise_scale]  
    res = pd.DataFrame(dict(zip(labels, values)), index=[0])

    # datetime object containing current date and time
    now = datetime.now() 

    df = create_sim_data_base(absolute_treatment_effect, num_orders, num_retailers, 
                              retailer_loc, retailer_scale, noise_loc, noise_scale) 
    
    file_name = './data/' + 'sim_data_' + now.strftime("%Y_%m_%d_%H%M%S") + '.csv'
    df.to_csv(file_name, index=False)
    res.to_csv('./data/' + 'data_param' + now.strftime("%Y_%m_%d_%H%M%S") + '.csv', index=False)
    
    return file_name


################################################################################################
def set_starting_value(data, treatment_variable, covariates, Y, absolute_mde, rejection_region, desired_power):

    treatment_variable = 'Treated'
    covariates = ['Mean_Retailer_Amount']
    Y = 'Initial_Order_Amount'

    covariates.append(treatment_variable)
    covariates.reverse()

    X = data[covariates]
    X = sm.add_constant(X)
    Y = data[Y]

    model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()
    Y2 = pd.DataFrame(model1.resid)
    Y2['Constant'] = 1 
    X2 = Y2['Constant']
    Y2 = Y2[[0]]
    model2 = sm.OLS(Y2.astype(float), X2.astype(float)).fit()
    se_sd = model2.bse[0] * np.sqrt(Y2.shape[0])
    effect_size = absolute_mde / se_sd
    recommended_n = int(sm.stats.tt_ind_solve_power(effect_size = effect_size, 
                        alpha = rejection_region, power = desired_power, 
                        alternative = 'larger'))
    return recommended_n


################################################################################################
def assess_power(data, 
                 candidate_n, 
                 treatment_variable, 
                 covariates, 
                 dependent_variable, 
                 rejection_region = 0.05, 
                 sims = 100):
    
    """Takes a sample size runs OLS simulations to determine effective statistical power.

    Keyword arguments:
    data               -- [Pandas dataframe] The sample data
    candidate_n        -- [int] The proposed sample size
    treatment_variable -- [string] The treatment variable passed to the OLS model
    covariates         -- [list of strings] A list of variable names - the 'X' argument for StatsModel
    dependent_variable -- [string] The dependent 'outcome' variable - the 'Y' argument for StatsModel
    rejection_region   -- [Float] The rejection/alpha region and delimiter of statistical significance
    sims               -- [int] The number of simulations to perform
    """ 
    
    if candidate_n > len(data):
        print("The proposed size of the sub-sample exceeded the size of the parent sample.")
    else:
        print("Estimating the effective power of n = {:,} using {} simulations.".format(candidate_n, sims))
        covariates.append(treatment_variable)
        covariates.reverse()
        results = []

        i = 0
        while i <= sims:
            subsample = data.sample(candidate_n)
            x = subsample[covariates]
            x = sm.add_constant(x)
            y = subsample[dependent_variable]
            model = sm.OLS(y.astype(float), x.astype(float)).fit()
            if model.pvalues[1] < rejection_region:
                results.append(1)
            else:
                results.append(0)
            i = i + 1
        power = mean(results)
        print("The effective power of sample size " + \
              "n = {:,} is {}%.".format(candidate_n, round(power*100,2)))
        return power

################################################################################################
def direct_search(data, 
                  starting_value, 
                  treatment_variable, 
                  covariates, 
                  dependent_variable, 
                  sims              = 250,
                  precision         = 0.05, 
                  rejection_region  = 0.05, 
                  desired_power     = 0.8,
                  prior_sims        = 0
                  ):
    
    """Takes a starting sample size and recommends retention (0), or and increase/decrease in n (1, -1).

    Keyword arguments:
    data               -- [Pandas dataframe] The sample data
    starting_value     -- [int] The initially proposed sample size
    treatment_variable -- [string] The treatment variable passed to the OLS model
    covariates         -- [list of strings] A list of variable names - the 'X' argument for StatsModel
    dependent_variable -- [string] The dependent 'outcome' variable - the 'Y' argument for StatsModel
    verification_sims  -- [int] The initial number of simulations used to verify if the starting_value
                          is overpowered or underpowered
    precision          -- [Float] How close the estimated power must be with respect to the target power
    rejection_region   -- [Float] The rejection/alpha region and delimiter of statistical significance
    desired_power      -- [Float] Desired power in percentage points 
    """ 
    
    print("Determining direction of search.")
    print("Starting sample size recommendation is {:,}.".format(starting_value))
    power = assess_power(data, starting_value, treatment_variable, covariates, 
                         dependent_variable, rejection_region, sims)
    print("Starting power is {}%.".format(round(power*100, 2)))
    if abs(power - desired_power) < precision:
        print("Sample size of n = {:,} meets the specified level of precision.".format(starting_value))
        return 0, power, sims
    elif power > desired_power:
        print("Sample size of n = {:,} is over-powered.".format(starting_value))
        return -1, power, sims
    elif power < desired_power:
        print("Sample size of n = {:,} is under-powered.".format(starting_value))
        return 1, power, sims + prior_sims

################################################################################################    
def create_initial_search_delimiters(starting_value, 
                                     direction, 
                                     search_orders, 
                                     desired_power):
    if direction == 0:
        print("Suitable sample size found.")
    elif direction == -1:
        print("Proposed sample size of {:,} exceeds ".format(starting_value) +
              "desired power of {}%.".format(round(desired_power*100)))
        lb = int(starting_value/(10*search_orders))
        print("The lower-bound for the proposed sample size search is {:,}.".format(lb))
        ub = starting_value
        print("The upper-bound for the proposed sample size search is {:,}.".format(ub))  
    elif direction == 1:
        print("Proposed sample size of {:,} fails to meet ".format(starting_value) +
              "desired power of {}%.".format(round(desired_power*100)))
        lb = starting_value
        print("The lower-bound for the proposed sample size search is {:,}.".format(lb))
        ub = int(starting_value*pow(10, search_orders))
        print("The upper-bound for the proposed sample size search is {:,}.".format(ub))
    return lb, ub    


################################################################################################
def create_range(lb, ub, points_per_iteration):
    
    """Creates n candidate sample sizes between lower-bound (lb) and upper-bound (ub)

    Keyword arguments:
    lb                   -- [Int] Lower-bound of potential sample sizes 
    ub                   -- [Int] Upper-bound of potential sample sizes  
    points_per_iteration -- [Int] Number of sample size candidates to be created
    """ 
    
    candidates = []
    points_per_iteration = 5
    for i in np.linspace(lb, ub, points_per_iteration + 1):
        candidates.append(int(i))
    candidates = candidates[1:points_per_iteration]
    candidates.append(int(candidates[points_per_iteration-2] + \
                         (ub - candidates[points_per_iteration-2])/2))
    print("{} candidate sample sizes were recommended within the range ".format(len(candidates)) +\
          "{:,} to {:,}.".format(lb, ub))
    return candidates


################################################################################################
def results_scrub(final_power, desired_power, candidates, results):
    cleaned_results = []
    if final_power > desired_power:
        for (key, value) in dict(zip(candidates, results)).items():
            if value < desired_power:
                cleaned_results.append((key, value))
    else:
        for (key, value) in dict(zip(candidates, results)).items():
            if value > desired_power:
                cleaned_results.append((key, value))
    deltas = []
    for i in cleaned_results:
        deltas.append(abs(i[1] - desired_power))
   
    runner_up = cleaned_results[deltas.index(min(deltas))]
    return runner_up[0], runner_up[1]

################################################################################################
def interpolate_recommendation(data, 
                               candidates, 
                               treatment_variable, 
                               covariates, 
                               dependent_variable, 
                               model = 'isotonic', 
                               precison = 0.01, 
                               desired_power = 0.8, 
                               rejection_region = 0.05, 
                               sims = 250,
                               sims_used = 0):
    
    if model == 'isotonic':
        print('Proceeding with interpolation via isotonic regression.')
    elif model == 'exponential_cdf':
        print('Proceeding with interpolation using the CDF of the exponential distribution.')
        def exp_cdf(x, a, b, c):
            return a * np.exp(-b * x) + c

    results = []
    for i in candidates:
        power = assess_power(data, i, treatment_variable, 
                             covariates, dependent_variable, rejection_region, sims)
        sims_used += sims
        delta = abs(desired_power - power)
        if delta < precison:
            print("Recommended sample size was attained after {:,} simulations with ".format(sims_used) +\
                  "(n = {:,}, power = {}%).".format(i, round(power*100, 2)))
            print("The results meet the required precision of +/- {} ".format(round(precison*100,1)) +\
                  "points of statistical power.")
            return i, power, i, power, sims_used
        else:
            results.append(power)

    if model == 'isotonic':  
        try:
            if min(results) > desired_power:
                print("Specified range for isotonic regression failed to capture desired power.")
                print("Now supplementing range.")
                supp_n = int(min(candidates)/2)
                supp_power = assess_power(data, supp_n, treatment_variable, 
                                     covariates, dependent_variable, rejection_region, sims)
                sims_used += sims
                delta = abs(supp_power - power)
                if delta < precison:
                    print("Recommended sample size was attained after {:,} simulations with ".format(sims_used) +\
                          "(n = {:,}, power = {}%).".format(supp_n, round(supp_power*100, 2)))
                    print("The results meet the required precision of +/- {} ".format(round(precison*100,1)) +\
                          "points of statistical power.")
                    return supp_n, supp_power, supp_n, supp_power, sims_used
                else:
                    results.append(supp_power)
                    candidates.append(supp_n)
                    
            iso_reg = IsotonicRegression().fit(results, candidates)
            recommendation = int(iso_reg.predict([desired_power]))
            
        except:
            print("scikit-learn IsotonicRegression.predict() failed to model the inputted values.")
            
    elif model == 'exponential_cdf':
        try:
            eta_y = np.asarray(results)
            cdf_x = np.asarray(candidates)
            popt, pcov = curve_fit(exp_cdf, eta_y, cdf_x)
            recommendation = int(popt[0]*np.exp(-(popt[1]) * (desired_power)) + popt[2])
        except:
            print("SciPy curve_fit() failed to parameterize the inputted values.")
            print("This can happen when the inputted power estimates have high variance.")
            print("Consider increasing the number of simulations to reduce variance.")

    print("A sample size of {:,} was recommended ".format(recommendation) +\
          "using {:,} simulations.".format(sims_used))
    final_power = assess_power(data, recommendation , treatment_variable, 
                               covariates, dependent_variable, 
                               rejection_region, sims)
    sims_used += sims
    delta = abs(desired_power - final_power)
    if delta < precison:
        print("Recommended sample size was attained after {:,} simulations with ".format(sims_used) +\
              "(n = {}, power = {}%).".format(recommendation, round(final_power*100, 2)))
        print("The results meet the required precision of +/- {} ".format(round(precison*100,1)) +\
              "points of statistical power.")
        return recommendation, final_power, recommendation, final_power, sims_used
    else:
        print("Results failed to meet required precision of +/- {} ".format(round(precison*100,1)) +\
              "points of statistical power.")
        runner_up_sample, runner_up_power = results_scrub(final_power, 
                                                          desired_power, 
                                                          candidates, 
                                                          results)
        if runner_up_sample > recommendation:
            return recommendation, final_power, runner_up_sample, runner_up_power, sims_used
        else:
            return runner_up_sample, runner_up_power, recommendation, final_power, sims_used



################################################################################################
def create_interpolated_recommendation(
                                      data, 
                                      candidates, 
                                      treatment_variable, 
                                      covariates, 
                                      dependent_variable, 
                                      model = 'isotonic', 
                                      precison = 0.01, 
                                      desired_power = 0.8, 
                                      rejection_region = 0.05, 
                                      points_per_iteration = 5,
                                      sims = 250,
                                      prior_sims = 0
                                      ):

    a,b,c,d,e = interpolate_recommendation(data, candidates, treatment_variable, covariates, 
                                           dependent_variable, model, precison, desired_power, 
                                           rejection_region, sims, prior_sims)
    while a != c:
        candidates = create_range(a, c, points_per_iteration)
        a,b,c,d,e = interpolate_recommendation(data, candidates, treatment_variable, covariates, 
                                               dependent_variable, model, precison, desired_power, 
                                               rejection_region, sims, sims_used = e)
    print("Interpolation concluded after {:,} simulations with \na ".format(e) +\
          "recommended sample size of {:,} with {}% statistical power".format(a, round(b*100, 2)))
    return a, b, e

################################################################################################
def derive_sample_size_recommendation(data, treatment_variable, covariates, 
                                      dependent_variable, absolute_mde, 
                                      rejection_region, desired_power, 
                                      search_orders, points_per_iteration, 
                                      sims, precison, model):    
    start = time.time()

    starting_value = set_starting_value(data, treatment_variable, covariates, dependent_variable, 
                                        absolute_mde, rejection_region, desired_power)

    direction, starting_power, prior_sims = direct_search(data, starting_value, treatment_variable, 
                                                          covariates, dependent_variable, sims,
                                                          precison, rejection_region, desired_power)

    lb, ub = create_initial_search_delimiters(starting_value, direction, search_orders, desired_power)

    candidates = create_range(lb, ub, points_per_iteration)

    n, n_power, sims_used = create_interpolated_recommendation(data, candidates, treatment_variable, 
                                                               covariates, dependent_variable, 
                                                               model, precison, desired_power, 
                                                               rejection_region, points_per_iteration,
                                                               sims, prior_sims)
    end = time.time()
    time_elapsed = end - start
    time_min = int(time_elapsed/60)
    time_sec = int(time_elapsed - (time_min*60))

    print("Power estimation took {} minutes and {} seconds.".format(time_min, time_sec))
    return n, n_power, sims_used, int(time_elapsed)