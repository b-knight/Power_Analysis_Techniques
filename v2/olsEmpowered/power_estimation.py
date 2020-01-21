import os
import time
import numpy as np
import pandas as pd
from statistics import mean 
import statsmodels.api as sm
from olsEmpowered import power_estimation


def save_results(df, sim_data_ob):

    import os
    import pandas as pd

    current_dir = os.getcwd() 
    if not os.path.exists(current_dir + '/results'):
        os.makedirs(current_dir + '/results') 
    df['file_name'] = sim_data_ob.data_file_name
    src  = str(type(sim_data_ob)).split('.')[1]
    if src == 'binary_search':
        if sim_data_ob.informed == 1:
            model_type = 'informed_' + str(type(sim_data_ob)).split('.')[1]
            df['source'] = model_type
            
        else:
            model_type = 'naive_' + str(type(sim_data_ob)).split('.')[1]
            df['source'] = model_type
    elif src == 'isotonic':
        model_type = 'isotonic'
        df['source'] = str(type(sim_data_ob)).split('.')[1]
        
    file_loc = current_dir + '/results/' + sim_data_ob.data_file_name[0:-4] + \
               '_' + model_type + '.csv'

    df.to_csv(file_loc, index = False)
    print("Saved the results from {}.".format(sim_data_ob.data_file_name[0:-4]))


class power_estimation:
    
    # constructor
    def __init__(self, sim_data_ob,
                 rejection_region = 0.05, 
                 desired_power    = 0.8,
                 precision        = 0.025,
                 covariates       = None):
        
        # set class variables
        self.rejection_region        = rejection_region
        self.desired_power           = desired_power
        self.precision               = precision
        
        # set super-class variables
        self.dv_name                 = sim_data_ob.dv_name     
        self.dv_cardinality          = sim_data_ob.dv_cardinality  
        self.treatment_variable      = sim_data_ob.treatment_variable  
        self.absolute_effect_size    = sim_data_ob.absolute_effect_size    
        self.sample_size             = sim_data_ob.sample_size    
        self.stats                   = sim_data_ob.stats 
        self.data                    = sim_data_ob.data
        self.data_file_name          = sim_data_ob.data_file_name
        self.data_file_location      = sim_data_ob.data_file_location
        self.meta_data_file_name     = sim_data_ob.meta_data_file_name
        self.meta_data_file_location = sim_data_ob.meta_data_file_location
        self.rsquared                = sim_data_ob.rsquared
        self.rsquared_adj            = sim_data_ob.rsquared_adj 
        if sim_data_ob.covariates is not None:
            self.covariates          = sim_data_ob.covariates       
            
        def set_starting_value(sim_data_ob):
        
            try:
                covariates = list(sim_data_ob.covariates.keys())
                covariates.append(sim_data_ob.treatment_variable)
                covariates.reverse()
                X = sim_data_ob.data[covariates]
            except:
                X = sim_data_ob.data[self.treatment_variable]    

            X = sm.add_constant(X)
            Y = sim_data_ob.data[self.dv_name]

            model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()

            Y2 = pd.DataFrame(model1.resid)
            Y2['Constant'] = 1 
            X2 = Y2['Constant']
            Y2 = Y2[[0]]
            model2 = sm.OLS(Y2.astype(float), X2.astype(float)).fit()

            se_sd = model2.bse[0] * np.sqrt(Y2.shape[0])
            effect_size = sim_data_ob.absolute_effect_size / se_sd
            recommended_n = int(sm.stats.tt_ind_solve_power(effect_size = effect_size, 
                                alpha = rejection_region, 
                                power = desired_power, 
                                alternative = 'larger'))
            return recommended_n

        self.starting_value = set_starting_value(sim_data_ob)

        
    def assess_power(self, candidate_n, sims):

        if candidate_n > self.sample_size:
            print("The proposed size of the sub-sample exceeded the size of the parent sample.")
        else:
            print("Estimating the effective power of n = {:,} using {:,} simulations.".format(candidate_n, sims))
            start = time.time()
            if hasattr(self, 'covariates'):
                covariates = list(self.covariates.keys())
                covariates.append(self.treatment_variable)
                covariates.reverse()
            else:
                covariates = [self.treatment_variable]
                
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
        