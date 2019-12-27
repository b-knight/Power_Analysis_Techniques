import os
import time
import numpy as np
import pandas as pd
from statistics import mean 
import statsmodels.api as sm


class interpolation:
    
    # constructor
    def __init__(self, sim_data_ob,
                 rejection_region = 0.05, 
                 desired_power    = 0.8,
                 precision        = 0.025,
                 search_orders    = 1):

        # set class variables
        self.rejection_region     = rejection_region
        self.desired_power        = desired_power
        self.precision            = precision
        self.search_orders        = search_orders
        self.dv_name              = sim_data_ob.dv_name     
        self.dv_cardinality       = sim_data_ob.dv_cardinality  
        self.treatment_variable   = sim_data_ob.treatment_variable  
        self.absolute_effect_size = sim_data_ob.absolute_effect_size    
        self.sample_size          = sim_data_ob.sample_size  
        self.covariates           = sim_data_ob.covariates       
        self.data                 = sim_data_ob.data
    
      
    def set_starting_value(self):
        
        covariates = list(self.covariates.keys())
        covariates.append(self.treatment_variable)
        covariates.reverse()

        X = self.data[covariates]
        X = sm.add_constant(X)
        Y = self.data[self.dv_name]
        
        model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()

        Y2 = pd.DataFrame(model1.resid)
        Y2['Constant'] = 1 
        X2 = Y2['Constant']
        Y2 = Y2[[0]]
        model2 = sm.OLS(Y2.astype(float), X2.astype(float)).fit()

        se_sd = model2.bse[0] * np.sqrt(Y2.shape[0])
        effect_size = self.absolute_effect_size / se_sd
        recommended_n = int(sm.stats.tt_ind_solve_power(effect_size = effect_size, 
                            alpha = self.rejection_region, 
                            power = self.desired_power, 
                            alternative = 'larger'))
        return recommended_n

    
    def set_upper_bound(self):
        n = interpolation.set_starting_value(self)
        return int(n*(pow(10,self.search_orders)))
    
    
    def set_lower_bound(self):
        n = interpolation.set_starting_value(self)
        return int(n/(pow(10,self.search_orders)))
        
        
    def assess_power(self, candidate_n, sims):

        if candidate_n > self.sample_size:
            print("The proposed size of the sub-sample exceeded the size of the parent sample.")
        else:
            print("Estimating the effective power of n = {:,} using {} simulations.".format(candidate_n, sims))
            start = time.time()
            covariates = list(self.covariates.keys())
            covariates.append(self.treatment_variable)
            covariates.reverse()
            results = []

            i = 0
            while i <= sims:
                subsample = self.data.sample(candidate_n)
                x = subsample[covariates]
                x = sm.add_constant(x)
                y = subsample[self.dv_name]
                model = sm.OLS(y.astype(float), x.astype(float)).fit()
                if model.pvalues[1] < self.rejection_region:
                    results.append(1)
                else:
                    results.append(0)
                i = i + 1
            
            power = mean(results)
            
            end = time.time()
            time_elapsed = end - start
            
            print("The effective power of sample size " + \
                  "n = {:,} is {}%.".format(candidate_n, round(power*100,2)))
           
            return power, sims, time_elapsed       
