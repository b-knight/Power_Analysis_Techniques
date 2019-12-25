import numpy as np
import pandas as pd
import statsmodels.api as sm

class isotonic:
    
    # constructor
    def __init__(self, sim_data_ob, 
                 rejection_region = 0.05, 
                 desired_power = 0.8):

        # set class variables
        self.rejection_region     = rejection_region
        self.desired_power        = desired_power
        self.dv_name              = sim_data_ob.dv_name     
        self.dv_cardinality       = sim_data_ob.dv_cardinality  
        self.treatment_variable   = sim_data_ob.treatment_variable  
        self.absolute_effect_size = sim_data_ob.absolute_effect_size    
        self.sample_size          = sim_data_ob.sample_size  
        self.covariates           = sim_data_ob.covariates       
        self.data                 = sim_data_ob.data
    
    def set_starting_value(isotonic):

        covariates = list(isotonic.covariates.keys())
        covariates.append(isotonic.treatment_variable)
        covariates.reverse()

        X = isotonic.data[covariates]
        X = sm.add_constant(X)
        Y = isotonic.data[isotonic.dv_name]
        model1 = sm.OLS(Y.astype(float), X.astype(float)).fit()
        print(model1)
        
        Y2 = pd.DataFrame(model1.resid)
        Y2['Constant'] = 1 
        X2 = Y2['Constant']
        Y2 = Y2[[0]]
        model2 = sm.OLS(Y2.astype(float), X2.astype(float)).fit()
        print(model2)
        
        se_sd = model2.bse[0] * np.sqrt(Y2.shape[0])
        effect_size = isotonic.absolute_effect_size / se_sd
        recommended_n = int(sm.stats.tt_ind_solve_power(effect_size = effect_size, 
                            alpha = isotonic.rejection_region, 
                            power = isotonic.desired_power, 
                            alternative = 'larger'))
        return recommended_n

                 