import pandas as pd
from olsEmpowered import power_estimation

class binary_search(power_estimation.power_estimation):

    # constructor
    def __init__(self, power_estimation,
                 sims_per_point   = 200,
                 search_orders    = 1,
                 informed         = 1):
        
        # set class variables
        self.sims_per_point          = sims_per_point
        self.search_orders           = search_orders
        self.informed                = informed
        
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
        
        
    def preliminary_screen(self):
        
        result_dict = {}      
        
        if self.informed == 1:
            ub      = self.starting_value*pow(10,self.search_orders) 
            mid     = self.starting_value  
            print("Binary search commenced, centered on n = " +
                  "{:,} informed by residual variance.".format(mid))
        else:
            ub      = len(self.data)
            mid     = int(len(self.data)/2)
            print("Binary search commenced, naively centered on n = {:,}.".format(mid))
            
        results     = []
        candidates  = []
        
        sims_used   = 0
        time_taken  = 0

        for i in [mid, ub]:
            power_est, sims, secs = self.assess_power(i, self.sims_per_point)
            results.append(power_est)
            candidates.append(i)
            sims_used += sims
            time_taken += secs
            
            if abs(power_est - self.desired_power) < self.precision:
                candidates.sort()
                result_dict.update({'candidates': candidates})
                results.sort()
                result_dict.update({'power': results})
                result_dict.update({'sims_used': sims_used})
                result_dict.update({'seconds_used': time_taken})
                result_dict.update({'status': 0})
                print("Binary search captured the desired power of " +
                      "{} (n = {:,}).".format(self.desired_power, i))
                return result_dict      
            
            elif (i == mid) & (power_est > self.desired_power):
                result_dict.update({'candidates': [0, mid]})
                result_dict.update({'power': [0, power_est]})
                result_dict.update({'sims_used': sims_used})
                result_dict.update({'seconds_used': time_taken})
                result_dict.update({'status': 1})
                print("Binary search suggested the region of " +
                      "n = ({:,}, {:,}).".format(0, mid))
                return result_dict
            
            elif (i == ub) & (power_est > self.desired_power):
                result_dict.update({'candidates': candidates})
                result_dict.update({'power': results})
                result_dict.update({'sims_used': sims_used})
                result_dict.update({'seconds_used': time_taken})
                result_dict.update({'status': -1})
                print("Binary search suggested the region of " +
                      "n = ({:,}, {:,}).".format(mid, ub))
                return result_dict
            
            elif (i == ub) & (power_est < self.desired_power):
                print("Simulation data of length {:,} exhausted. ".format(ub))
                print("Try using more simulation data (max power " +\
                      "attained = {}%).".format(round(max(results)*100)))
                return None
                  
            
    def binary_parent(self, input_dict):

        results_dict = {}
        
        parent_candidates   = input_dict.get('candidates')
        parent_results      = input_dict.get('power')
        parent_sims_used    = input_dict.get('sims_used')
        parent_seconds_used = input_dict.get('seconds_used')

        current_n = 0
        current_p = 0

        def binary_child(recursion_candidates, recursion_results):
            
            nonlocal parent_candidates   
            nonlocal parent_results      
            nonlocal parent_sims_used    
            nonlocal parent_seconds_used 
            nonlocal current_n 
            nonlocal current_p          
            
            lb_n      = recursion_candidates[0]
            lb_power  = recursion_results[0]
            ub_n      = recursion_candidates[1]     
            ub_power  = recursion_results[1]
            current_n = int(lb_n + ((ub_n - lb_n)/2))
            parent_candidates.append(current_n)

            print("Now assessing the value {:,} ".format(current_n) +
                  "within the range n = ({:,}, {:,}), ".format(lb_n, ub_n) +
                  "and power = ({}%, {}%).".format(round(lb_power*100,2), 
                                                   round(ub_power*100,2)))
            current_p, sims_executed, seconds_elapsed = self.assess_power(current_n, 
                                                                          self.sims_per_point)
            parent_results.append(current_p) 
            parent_sims_used    += sims_executed
            parent_seconds_used += seconds_elapsed

            if current_p > self.desired_power:
                recursion_candidates = [lb_n, current_n]
                recursion_results    = [lb_power, current_p]    

            else:
                recursion_candidates = [current_n, ub_n]
                recursion_results    = [current_p, ub_power]   
                
            recursion_candidates.sort()  
            recursion_results.sort()
                
            return recursion_candidates, recursion_results

        recursion_candidates = parent_candidates   
        recursion_results    = parent_results                     
                
        while abs(current_p - self.desired_power) > self.precision:
            recursion_candidates, recursion_results = binary_child(recursion_candidates, 
                                                                   recursion_results)
        results_dict.update({'candidates': parent_candidates})
        results_dict.update({'power': parent_results})
        results_dict.update({'sims_used': parent_sims_used})
        results_dict.update({'seconds_used': parent_seconds_used})
        results_dict.update({'status': 0})
        
        return current_n, current_p, results_dict
        
            
    def combine_dfs(self, current_n, current_p, parent_dict, recursion_dict):
        prior_sims     = parent_dict.get('sims_used')
        prior_time     = parent_dict.get('seconds_used')
        recursive_sims = recursion_dict.get('sims_used')
        recursive_time = recursion_dict.get('seconds_used')
        combined_sims  = prior_sims + recursive_sims
        combined_time  = prior_time + recursive_time
        recursion_dict.update({'sims_used': combined_sims})
        recursion_dict.update({'seconds_used': combined_time})
        df = pd.DataFrame(recursion_dict)
        return current_n, current_p, df        
        

    def binary_search(self):
        preliminary_dict = self.preliminary_screen()
        current_n, current_p, recursion_dict = self.binary_parent(preliminary_dict)
        final_n, final_p, results = self.combine_dfs(current_n, current_p, 
                                                     preliminary_dict, recursion_dict)
        return final_n, final_p, results
    
    
    
    
    
    
    
    
    
    
    
    
    
        
