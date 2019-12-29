import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from olsEmpowered import interpolation


class logarithmic(interpolation.interpolation):

    # constructor
    def __init__(self, sim_data_ob,
                 rejection_region = 0.05,
                 desired_power    = 0.8,
                 precision        = 0.025,
                 search_orders    = 1,
                 sims_per_point   = 200):
        
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
        self.sims_per_point       = sims_per_point
        
        
    def logarithmic_interpolation(self):

        results_dict      = {}
        parent_candidates = []               
        parent_results    = []
              
        parent_candidates.append(self.set_lower_bound())
        parent_candidates.append(self.set_starting_value())
        parent_candidates.append(self.set_upper_bound())
        
        parent_sims_used    = 0
        parent_seconds_used = 0
        
        for i in parent_candidates:
            power_est, sims_used, secs_taken = self.assess_power(i, self.sims_per_point)
            parent_results.append(power_est)
            parent_sims_used    += sims_used
            parent_seconds_used += secs_taken
            
            
        left_end = 0   
        while parent_results[left_end] <= 0.10:
            print("increasing lb")
            j = int(parent_candidates[left_end] + 
                    ((parent_candidates[left_end + 1] - parent_candidates[left_end])/2))
            parent_candidates.append(j)
            power_est, sims_used, secs_taken = self.assess_power(j, self.sims_per_point)
            parent_results.append(power_est)
            parent_sims_used    += sims_used
            parent_seconds_used += secs_taken  
            parent_candidates.sort() 
            parent_results.sort() 
            left_end += 1
        
        right_end = -1
        while parent_results[right_end] >= 0.90:
            print("decreasing ub")
            k = int(parent_candidates[right_end] - 
                    ((parent_candidates[right_end] - parent_candidates[right_end - 1])/2))
            parent_candidates.append(k)
            power_est, sims_used, secs_taken = self.assess_power(k, self.sims_per_point)
            parent_results.append(power_est)
            parent_sims_used    += sims_used
            parent_seconds_used += secs_taken
            parent_candidates.sort() 
            parent_results.sort() 
            right_end -= 1

        current_n = 0
        current_p = 0

        
        def func(x, a, b):
            return a*np.log(x)+ b
        
#         def func(x,a,b,c):
#             return a * np.exp(-b * x) + c
        
        def logarithmic_child(logarithmic_candidates, 
                              logarithmic_results):

            nonlocal current_n 
            nonlocal current_p
            nonlocal parent_candidates
            nonlocal parent_results 
            nonlocal parent_sims_used
            nonlocal parent_seconds_used
                        
            popt, pcov = curve_fit(func, logarithmic_results, 
                                   logarithmic_candidates)
            current_n = int(func(0.8, *popt))
            parent_candidates.append(current_n)
            current_p, sims_used, secs_taken = self.assess_power(current_n, 
                                                                 self.sims_per_point)
            parent_results.append(current_p)
            parent_sims_used    += sims_used
            parent_seconds_used += secs_taken

            return logarithmic_candidates, logarithmic_results  

        logarithmic_candidates = parent_candidates
        logarithmic_results    = parent_results  

        while abs(current_p - self.desired_power) > self.precision:
            logarithmic_candidates, logarithmic_results = logarithmic_child(logarithmic_candidates, 
                                                                            logarithmic_results)

        results_dict.update({'candidates': logarithmic_candidates})
        results_dict.update({'power': logarithmic_results})
        results_dict.update({'sims_used': parent_sims_used})
        results_dict.update({'seconds_used': parent_seconds_used})
        results_dict.update({'status': 0})

        return current_n, current_p,  pd.DataFrame(results_dict)         


