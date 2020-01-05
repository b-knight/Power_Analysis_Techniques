import numpy as np
import pandas as pd
from olsEmpowered import power_estimation
from sklearn.isotonic import IsotonicRegression

class isotonic(power_estimation.power_estimation):

    # constructor
    def __init__(self, power_estimation,
                 sims_per_point = 200,
                ):
        
        # set class variables
        self.sims_per_point          = sims_per_point
        
        self.rejection_region        = power_estimation.rejection_region
        self.desired_power           = power_estimation.desired_power
        self.precision               = power_estimation.precision
        self.dv_name                 = power_estimation.dv_name     
        self.dv_cardinality          = power_estimation.dv_cardinality  
        self.treatment_variable      = power_estimation.treatment_variable  
        self.absolute_effect_size    = power_estimation.absolute_effect_size    
        self.sample_size             = power_estimation.sample_size  
        self.stats                   = power_estimation.stats 
        self.data                    = power_estimation.data
        self.data_file_name          = power_estimation.data_file_name
        self.data_file_location      = power_estimation.data_file_location
        self.meta_data_file_name     = power_estimation.meta_data_file_name
        self.meta_data_file_location = power_estimation.meta_data_file_location
        self.rsquared                = power_estimation.rsquared
        self.rsquared_adj            = power_estimation.rsquared_adj   
        self.starting_value          = power_estimation.starting_value
        
        if hasattr(power_estimation, 'covariates'):
            self.covariates          = power_estimation.covariates  

 
    def isotonic_interpolation(self):
#########################################################################    
        parent_candidates   = []               
        parent_results      = []
        parent_sims_used    = 0
        parent_seconds_used = 0
        
        # specify results packaging and return method
        def return_results(n, power_estimation, 
                           parent_candidates, parent_results,
                           parent_sims_used, parent_seconds_used):
        
            results_dict = {}
            results_dict.update({'candidates': parent_candidates})
            results_dict.update({'power': parent_results})
            results_dict.update({'sims_used': parent_sims_used})
            results_dict.update({'seconds_used': parent_seconds_used})
            results_dict.update({'status': 0})            
            return n, power_estimation, pd.DataFrame(results_dict)        
        
        # assess starting value
        est_pow, sims_used, secs_taken = self.assess_power(self.starting_value, 
                                                           self.sims_per_point)
        parent_candidates.append(self.starting_value)
        parent_results.append(est_pow)
        parent_sims_used    += sims_used
        parent_seconds_used += secs_taken
        delta = abs(est_pow - self.desired_power)
          
        # return results if starting value fulfills requirements 
        if delta < self.precision:  
            a, b, c = return_results(self.starting_value, est_pow, 
                                     parent_candidates, parent_results,
                                     parent_sims_used, parent_seconds_used)
            return a, b, c
        
        
        elif (delta > self.precision) & (est_pow < self.desired_power):
            current_n  = self.starting_value
            current_p  = est_pow
            increments = 1
            while current_p < self.desired_power:
                current_n = int(current_n + \
                              ((self.desired_power - current_p)**2)*(current_n*pow(10,increments)))
                print("An upper-bound of n = {:,} was specified.".format(current_n))
                est_pow, sims_used, secs_taken = self.assess_power(current_n, 
                                                                   self.sims_per_point)
                parent_candidates.append(current_n)
                parent_results.append(est_pow)
                parent_sims_used    += sims_used
                parent_seconds_used += secs_taken  
                current_p = est_pow
                delta = abs(est_pow - self.desired_power)
                if delta < self.precision: 
                    a, b, c = return_results(current_n, current_p, 
                                             parent_candidates, parent_results,
                                             parent_sims_used, parent_seconds_used)
                    return a, b, c   
                increments += 1
                    
                    
        elif (delta > self.precision) & (est_pow > self.desired_power): 
            current_n = self.starting_value
            current_p = est_pow
            while current_p > self.desired_power:
                current_n = int(self.starting_value*((1-(est_pow-self.desired_power))^2))
                print("A lower-bound of n = {:,} was specified.".format(current_n))
                est_pow, sims_used, secs_taken = self.assess_power(current_n, 
                                                                   self.sims_per_point)
                parent_candidates.append(current_n)
                parent_results.append(est_pow)
                parent_sims_used    += sims_used
                parent_seconds_used += secs_taken  
                current_p = est_pow
                delta = abs(est_pow - self.desired_power)
                if delta < self.precision: 
                    a, b, c = return_results(current_n, current_p,
                                             parent_candidates, parent_results,
                                             parent_sims_used, parent_seconds_used)
                    return a, b, c                    
            
        parent_candidates.sort() 
        parent_results.sort() 
        
        if delta < self.precision:  
            a, b, c = return_results(current_n, current_p,
                                     parent_candidates, parent_results,
                                     parent_sims_used, parent_seconds_used)
            return a, b, c
        
        else:
            def isotonic_child(iso_candidates, iso_results):

                nonlocal current_n 
                nonlocal current_p
                nonlocal parent_candidates
                nonlocal parent_results 
                nonlocal parent_sims_used
                nonlocal parent_seconds_used

                iso_reg = IsotonicRegression().fit(iso_results, iso_candidates)
                current_n = int(iso_reg.predict([self.desired_power])) 
                parent_candidates.append(current_n)
                current_p, sims_used, secs_taken = self.assess_power(current_n, 
                                                                     self.sims_per_point)
                parent_results.append(current_p)
                parent_sims_used    += sims_used
                parent_seconds_used += secs_taken

                return iso_candidates, iso_results   

            iso_candidates = parent_candidates
            iso_results    = parent_results  

            while abs(current_p - self.desired_power) > self.precision:
                iso_candidates, iso_results = isotonic_child(iso_candidates, 
                                                             iso_results)
        
            a, b, c = return_results(current_n, current_p, 
                                     parent_candidates, parent_results,
                                     parent_sims_used, parent_seconds_used)
            return a, b, c    