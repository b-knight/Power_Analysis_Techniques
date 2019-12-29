import numpy as np
import pandas as pd
from olsEmpowered import interpolation
from sklearn.isotonic import IsotonicRegression

class isotonic(interpolation.interpolation):

    # constructor
    def __init__(self, sim_data_ob,
                 rejection_region = 0.05,
                 desired_power = 0.8,
                 precision = 0.025,
                 search_orders = 1,
                 points_per_iteration = 2, 
                 sims_per_point = 200):
        
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
        self.points_per_iteration = points_per_iteration
        self.sims_per_point       = sims_per_point
        
    
    def preliminary_screen(self):
  
        result_dict = {}
        candidates  = []
        results     = []
        sims_list   = []
        time_list   = []
        candidates.append(self.set_lower_bound())
        candidates.append(self.set_starting_value())
        candidates.append(self.set_upper_bound())
        
        for i in candidates:
            print("Now assessing n = {:,}.".format(i))
            power_est, sims_used, time_taken = self.assess_power(i, self.sims_per_point)
            results.append(power_est)
            sims_list.append(sims_used)
            time_list.append(time_taken)
            
            if abs(power_est - self.desired_power) < self.precision:
                total_sims = sum(sims_list)
                total_time = sum(time_list)
                print("Desired power attained with n = {:,} ({}% power) ".format(i, power_est) + 
                      "in {} seconds with {} simulations.".format(total_time, total_sims))
                result_dict.update({'candidates': candidates})
                result_dict.update({'power': results})
                result_dict.update({'sims_used': total_sims})
                result_dict.update({'seconds_used': total_time})
                result_dict.update({'status': 0})
                return result_dict
                
        if min(results) > self.desired_power:
            print("Specified range for isotonic regression failed to capture desired power.")
            print("Now supplementing range.")
            supp_n = int(min(candidates)/2)
            supp_est, supp_sims_used, supp_time = self.assess_power(supp_n, self.sims_per_point)
            candidates.append(supp_n)
            candidates.reverse()
            results.append(supp_est)
            results.reverse()
            sims_list.append(sims_used)
            sims_list.reverse()
            time_list.append(time_taken)
            time_list.reverse()
        elif max(results) < self.desired_power:
            print("Specified range for isotonic regression failed to capture desired power.")
            print("Now supplementing range.")
            supp_n = int(max(candidates)*2)
            supp_est, supp_sims_used, supp_time = self.assess_power(supp_n, self.sims_per_point)
            candidates.append(supp_n)
            results.append(supp_est)
            sims_list.append(sims_used)
            time_list.append(time_taken)               

        # execute isotonic regression
        iso_reg = IsotonicRegression().fit(results, candidates)
        recommendation = int(iso_reg.predict([self.desired_power])) 
        final_est, final_sims, final_time = self.assess_power(recommendation, self.sims_per_point)
        candidates.append(recommendation)
        results.append(final_est)
        sims_list.append(final_sims)
        time_list.append(final_time)   
        total_sims = sum(sims_list)
        total_time = sum(time_list)
        
        result_dict.update({'candidates': candidates})
        result_dict.update({'power': results})
        result_dict.update({'sims_used': total_sims})
        result_dict.update({'seconds_used': total_time})
        
        if abs(final_est - self.desired_power) < self.precision:
            result_dict.update({'status': 0})
        elif final_est > (self.desired_power + self.precision):
            result_dict.update({'status': -1})
        elif final_est < (self.desired_power - self.precision):
            result_dict.update({'status': 1})
        return result_dict
            
            
    def establish_new_range(self, result_dict):
        working_df = pd.DataFrame(result_dict)
        pd.options.mode.chained_assignment = None
        working_df['delta'] = abs(working_df['power'] - self.desired_power)
        min_df = working_df[working_df['power'] < self.desired_power]
        min_df.sort_values(['delta', 'candidates'], 
                           inplace = True, ascending=[True, False])
        lb = min_df.iat[0,0]
        max_df = working_df[working_df['power'] > self.desired_power]
        max_df.sort_values(['delta', 'candidates'], 
                           inplace = True, ascending=[True, True])
        ub = max_df.iat[0,0]
        return lb, ub
    
    
    def create_range(self, lb, ub):
        """Creates n candidate sample sizes between lower-bound (lb) and upper-bound (ub)

        Keyword arguments:
        lb                   -- [Int] Lower-bound of potential sample sizes 
        ub                   -- [Int] Upper-bound of potential sample sizes  
        points_per_iteration -- [Int] Number of sample size candidates to be created
        """ 
        candidates = []
        for i in np.linspace(lb, ub, self.points_per_iteration + 1):
            candidates.append(int(i))
        candidates = candidates[1:self.points_per_iteration]
        candidates.append(int(candidates[self.points_per_iteration-2] + \
                             (ub - candidates[self.points_per_iteration-2])/2))
        print("{} candidate sample sizes were recommended within the range ".format(len(candidates)) +\
              "{:,} to {:,}.".format(int(lb), int(ub)))
        return candidates
    
    
    def recursive_method(self, result, new_range):

        result_dict = {}
        current_power = 0.0
        working_df = pd.DataFrame(result)[['candidates', 'power']]
        working_df.sort_values(by='candidates', inplace = True, ascending = True)
        sims_taken = pd.DataFrame(result).sims_used.unique()[0]
        time_taken = pd.DataFrame(result).seconds_used.unique()[0]

        for i in new_range:
            est_power, sims_used, time_elapsed = self.assess_power(i, 200)
            working_df.loc[len(working_df)] = [i, est_power]
            sims_taken += sims_used
            time_taken += time_elapsed
            if abs(est_power - self.desired_power) < self.precision:
                result_dict.update({'candidates': working_df.candidates.values})
                result_dict.update({'power': working_df.power.values})
                result_dict.update({'sims_used': sims_taken})
                result_dict.update({'seconds_used': time_taken})
                result_dict.update({'status': 0})
                print("Target power of {} found at n = {:,}.".format(self.desired_power, i))
                return result_dict, new_range

        iso_reg = IsotonicRegression().fit(working_df.power.values, 
                                           working_df.candidates.values)
        recommendation = int(iso_reg.predict([self.desired_power]))
        est_power, sims_used, time_elapsed = self.assess_power(recommendation, 200)
        working_df.loc[len(working_df)] = [recommendation, est_power]
        
        sims_taken += sims_used
        time_taken += time_elapsed

        result_dict.update({'candidates': working_df.candidates.values})
        result_dict.update({'power': working_df.power.values})
        result_dict.update({'sims_used': sims_taken})
        result_dict.update({'seconds_used': time_taken})


        if abs(est_power - self.desired_power) < self.precision:
            print("Target power of {} found at n = {:,}.".format(self.desired_power, 
                                                               recommendation))
            result_dict.update({'status': 0})
        elif est_power > (self.desired_power + self.precision):
            result_dict.update({'status': -1})
        elif est_power < (self.desired_power - self.precision):
            result_dict.update({'status': 1})

        lb, ub = self.establish_new_range(result_dict)
        new_range = self.create_range(lb, ub)
        return result_dict, new_range
    
    
    def isotonic_interpolation(self):
        print("Interpolating required sample size using isotonic regression.")
        status = -999
        a      = self.preliminary_screen()
        lb, ub = self.establish_new_range(a)
        b      = self.create_range(lb, ub)
        while status != 0:
            a, b   = self.recursive_method(a, b)
            status = pd.DataFrame(a).status.values[0]

        results = pd.DataFrame(a)
        working_results = results[  (results['power'] < (self.desired_power + self.precision))
                                  & (results['power'] > (self.desired_power - self.precision))]
        final_n = int(working_results.candidates.values[0])
        final_p = int(working_results.power.values[0])
        return final_n, final_p, results
    
    
