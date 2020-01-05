import os
import ast
import time
import json
import numpy as np
import pandas as pd
import random as rand
import statsmodels.api as sm
from datetime import datetime



##################################################################################
def create_covariate_dict(max_covariates, 
                          permissible_distributions,
                          range_of_normal_loc, 
                          range_of_normal_scale,
                          range_of_exponential_scale,
                          range_of_uniform,
                          range_of_betas):

    num_covariates = rand.randint(0, max_covariates)
    if num_covariates == 0:
        print("This D.G.P. has no covariates besides the treatment variable.")
        return None
    else:
        print("{} additional covariates will be created.".format(num_covariates))

        probs = list(np.linspace(0, 1, len(permissible_distributions)+1)[1:])
        probs.sort(reverse=True)
        c = list(zip(permissible_distributions,probs))
        c.append(('blank', 0.0))
        covar = []
        for i in range(0, num_covariates):
            covar.append(rand.random()) 
        results = []
        for i in covar:
            for j in list(range(len(c)-1)):
                if (i < c[j][1]) & (i > c[j+1][1]):
                    results.append((i, c[j]))
        variable_names = []
        for k in range(num_covariates): 
            variable_names.append('X' + str(k))

        distributions = []
        [distributions.append(item[1][0]) for item in results]
        initial_dict = dict(zip(variable_names, distributions))
        for key in initial_dict:
            if initial_dict[key] == 'normal':
                blank_list = ['normal']
                params = [rand.uniform(range_of_normal_loc[0], 
                                       range_of_normal_loc[1]),
                          rand.uniform(range_of_normal_scale[0], 
                                       range_of_normal_scale[1])
                         ]
                blank_list.append(params)
                blank_list.append(rand.uniform(range_of_betas[0], 
                                               range_of_betas[1]))
                initial_dict.update({key: blank_list})

            elif initial_dict[key] == 'exponential':
                blank_list = ['exponential']
                params = [rand.uniform(range_of_exponential_scale[0], 
                                       range_of_exponential_scale[1])]
                blank_list.append(params)
                blank_list.append(rand.uniform(range_of_betas[0], 
                                               range_of_betas[1]))
                initial_dict.update({key: blank_list})

            elif initial_dict[key] == 'uniform':
                blank_list = ['uniform']
                params = [rand.uniform(range_of_uniform[0], 
                                       range_of_uniform[1]),
                          rand.uniform(range_of_uniform[0], 
                                       range_of_uniform[1])]
                params.sort()
                blank_list.append(params)
                blank_list.append(rand.uniform(range_of_betas[0], 
                                               range_of_betas[1]))
                initial_dict.update({key: blank_list})

            else:
                print("Could not interpret the specified distribution.")
                return None

    return initial_dict

##################################################################################
def create_random_dgp(range_of_normal_loc, 
                      range_of_normal_scale,
                      range_of_exponential_scale,
                      range_of_uniform,
                      range_of_betas,
                      range_of_abs_mde,
                      range_of_noise_loc,
                      range_of_noise_scale,
                      sample_size,
                      max_covariates = 10, 
                      permissible_distributions = ['normal', 
                                                   'exponential', 
                                                   'uniform'],):

    my_dict = create_covariate_dict(max_covariates = max_covariates, 
                                       permissible_distributions = permissible_distributions,
                                       range_of_normal_loc = range_of_normal_loc, 
                                       range_of_normal_scale = range_of_normal_scale,
                                       range_of_exponential_scale = range_of_exponential_scale,
                                       range_of_uniform = range_of_uniform,
                                       range_of_betas = range_of_betas)

    sim_ob = sim_data(dv_name = 'Y', 
                      dv_cardinality = 'continuous', 
                      sample_size = sample_size, 
                      absolute_effect_size = rand.uniform(range_of_abs_mde[0], 
                                                          range_of_abs_mde[1]), 
                      noise_loc = rand.uniform(range_of_noise_loc[0], 
                                               range_of_noise_loc[1]), 
                      noise_scale = rand.uniform(range_of_noise_scale[0], 
                                                 range_of_noise_scale[1]),
                      covariates_dict = my_dict)

    new_ob = sim_data(data_path = sim_ob.data_file_location + 
                                  '/' + sim_ob.data_file_name, 
                      meta_data_path = sim_ob.meta_data_file_location + 
                                  '/' + sim_ob.meta_data_file_name)
    
    new_ob.data_file_name          = sim_ob.data_file_name
    new_ob.data_file_location      = sim_ob.data_file_location
    new_ob.meta_data_file_name     = sim_ob.meta_data_file_name
    new_ob.meta_data_file_location = sim_ob.meta_data_file_location
    
    return new_ob 

##################################################################################
def remove_data(sim_ob, drop_meta_data = False):
    
    print(sim_ob.data_file_location)
    
    data_path = sim_ob.data_file_location + \
              '/' + sim_ob.data_file_name

    meta_data_path = sim_ob.meta_data_file_location + \
                   '/' + sim_ob.meta_data_file_name
    try:
        os.remove(data_path)
        print("The file {} was deleted.".format(sim_ob.data_file_name))
    except Exception as e:
        print("Failed to data file.")
        print(str(e)[10:])
    if drop_meta_data is True:
        try:
            os.remove(meta_data_path)
            print("The the associated meta-data for file {} ".format(sim_ob.data_file_name) +
                  "was also deleted.")
        except Exception as e:
            print("Failed to delete meta-data file.")
            print(str(e)[10:])

##################################################################################            
            
class sim_data:

    # constructor
    def __init__(self,           
                 dv_name              = None, 
                 dv_cardinality       = None,
                 absolute_effect_size = None,
                 sample_size = None, covariates_dict = None,
                 noise_loc   = None, noise_scale     = None,
                 data_path   = None, meta_data_path  = None,
                 rsquared    = None, rsquared_adj    = None):
          
        # create data generating method
        def create_dataframe(absolute_effect_size, 
                             sample_size, 
                             noise_loc, 
                             noise_scale,
                             covariates_dict = None):
            data_dict = {}

            if covariates_dict is not None:
                for key in covariates_dict:
                    distribution    = covariates_dict.get(key)[0]
                    parameter_list  = covariates_dict.get(key)[1]
                    beta_weight     = covariates_dict.get(key)[2]
                    if distribution == 'normal':
                        try:
                            data = list(np.random.normal(loc   = parameter_list[0], 
                                                         scale = parameter_list[1], 
                                                         size  = sample_size))
                            data_dict.update({key: data})
                        except:
                            print("Failed to generate simulation data of type = 'normal'.")

                    elif distribution == 'exponential':
                        try:
                            data = list(np.random.exponential(scale = parameter_list, 
                                                              size  = sample_size))
                            data_dict.update({key: data})
                        except:
                            print("Failed to generate simulation data of type = 'exponential'.")

                    elif distribution == 'uniform':
                        try:
                            data = list(np.random.uniform(low  = parameter_list[0], 
                                                          high = parameter_list[1], 
                                                          size = sample_size))
                            data_dict.update({key: data})
                        except:
                            print("Failed to generate simulation data of type = 'uniform'.")
                    else:
                        print('Could not generate data from the specified distribution.')
                        return None

                    df = pd.DataFrame(data_dict) 
                 
            else:
                df = pd.DataFrame()

            # Determine treatment assignment
            treatment_probs = []
            for i in list(range(0, sample_size, 1)):
                treatment_probs.append(np.random.uniform())
            df['prob'] = treatment_probs
   
            df['treated'] = np.where(df['prob'] <= 0.5, 1, 0)
            df['absolute_effect_size'] = np.where(df['prob'] <= 0.5, 
                                         absolute_effect_size, 0.0)

            # add noise
            df['noise'] = list(np.random.normal(loc   = noise_loc, 
                                                scale = noise_scale, 
                                                size  = sample_size))    
            del df['prob']
                   
            return df

        # if a data_path is not provided, create the necessary simulation data
        if ((dv_name is not None) & (dv_cardinality is not None)
        & (sample_size is not None) 
        & (absolute_effect_size is not None)
        & (noise_loc is not None) & (noise_scale is not None)
        & (data_path == None) & (meta_data_path == None)):

            start = time.time()
            
            # create file name
            now = datetime.now() 
            data_file_name = now.strftime("sim_data_%Y_%m_%d_%H%M%S") + '.csv' 
            current_dir = os.getcwd() 

            # create data
            try:
                df = create_dataframe(absolute_effect_size, sample_size, 
                                      noise_loc, noise_scale, covariates_dict)
                if covariates_dict is not None:
                    command_str = "df['{}'] = ".format(dv_name)
                    for key in covariates_dict:
                        command_str += "df['{}']*({}) + ".format(key, covariates_dict.get(key)[2])
                    command_str += "df['absolute_effect_size'] + df['noise']"
                    exec(command_str)                
                    df['ID'] = list(range(1, sample_size + 1,1))
                    del df['absolute_effect_size']
                    labels_order = ['ID', dv_name, 'treated'] + list(covariates_dict.keys())
                    labels_order.append('noise')
                    df = df[labels_order]
                else:
                    command_str = "df['{}'] = df['absolute_effect_size'] + df['noise']".format(dv_name)
                    exec(command_str)  
                    del df['absolute_effect_size']
                    df['ID'] = list(range(1, sample_size + 1,1))
                    df = df[['ID', dv_name, 'treated', 'noise']]
                
                # establish DGP parameters
                IV = []
                if covariates_dict is not None:
 
                    for key in covariates_dict:
                        IV.append(key)
                    IV.append('treated')
                    IV.reverse()
                else:    
                    IV.append('treated')
   
                X = df[IV]
                X = sm.add_constant(X)
                Y = df[dv_name]
                
                print("Now performing OLS to infer DGP parameters.")
                
                model = sm.OLS(Y.astype(float), X.astype(float)).fit()


                # create log file
                items_to_log = {}
                items_to_log.update({'file_name': data_file_name})
                items_to_log.update({'dv_name': dv_name})
                items_to_log.update({'dv_cardinality': dv_cardinality})
                items_to_log.update({'treatment_variable': 'treated'})
                items_to_log.update({'absolute_effect_size': absolute_effect_size})
                items_to_log.update({'sample_size': sample_size})
                if covariates_dict is not None:
                    items_to_log.update({'covariates': covariates_dict})
                description = pd.DataFrame(df.describe()).to_dict()
                items_to_log.update({'noise_loc': noise_loc})
                items_to_log.update({'noise_scale': noise_scale})
                items_to_log.update({'stats': description})
                items_to_log.update({'rsquared': model.rsquared})
                items_to_log.update({'rsquared_adj': model.rsquared_adj})
                items_to_log.update({'data_file_name': data_file_name})
                items_to_log.update({'meta_data_file_name': data_file_name[0:-4] + "_log_file.txt"})
                items_to_log.update({'data_file_location': current_dir + "/data"})
                items_to_log.update({'meta_data_file_location': current_dir + "/data/log_files"})                
                
            
                # create dir
                if not os.path.exists(current_dir + '/data/'):
                    os.makedirs(current_dir + '/data/') 
                if not os.path.exists(current_dir + '/data/log_files/'):
                    os.makedirs(current_dir + '/data/log_files/') 

                # save data
                df.to_csv(current_dir + "/data/" + data_file_name, index = False)
                print("Simulation data was saved to: \n     {}.".format(current_dir + \
                      "/data/" + data_file_name))

                # save log file
                log_file_name = current_dir + "/data/log_files/" + data_file_name[0:-4] + "_log_file.txt"
                with open(log_file_name, 'w') as file:
                     file.write(json.dumps(items_to_log))
                print("Meta-data was saved to: \n     {}.".format(current_dir + "/data/log_files/" + \
                      data_file_name[0:-4] + "_log_file.txt"))

                # report time
                end = time.time()
                time_elapsed = end - start
                time_min = int(time_elapsed/60)
                time_sec = int(time_elapsed - (time_min*60))
                print("{} observations of simulation data created in {} ".format(sample_size, time_min)  + \
                      "minutes and {} seconds.".format(time_sec))
                          
                # set class variables
                self.dv_name                 = dv_name
                self.dv_cardinality          = dv_cardinality
                self.treatment_variable      = 'treated'
                self.absolute_effect_size    = absolute_effect_size
                self.sample_size             = sample_size
                if covariates_dict is not None:
                    self.covariates              = covariates_dict
                self.data_file_name          = data_file_name
                self.data_file_location      = current_dir + '/data'
                self.meta_data_file_name     = data_file_name[0:-4] + "_log_file.txt"
                self.meta_data_file_location = current_dir + '/data/log_files'
                self.noise_loc               = noise_loc
                self.noise_scale             = noise_scale
                self.stats                   = description
                self.data                    = df
                self.rsquared                = model.rsquared
                self.rsquared_adj            = model.rsquared_adj 
                                
            except:
                print("Failed to generate the simulation data as specified.")
                
        # if a data_path is present, read data instead of creating it
        elif (data_path != None) & (meta_data_path != None):
            print("Reconstituting data object from file.")
            try:
                df = pd.read_csv(data_path) 
                print("Successfully read in the .csv file specified at:" + 
                      "\n     {}.".format(data_path))
                try:
                    print(meta_data_path)
                    
                    meta_string = open(meta_data_path, 'r').read()
                    meta_data = ast.literal_eval(meta_string)

                    # set class variables
                    self.dv_name                 = meta_data.get('dv_name')
                    self.dv_cardinality          = meta_data.get('dv_cardinality')
                    self.treatment_variable      = meta_data.get('treatment_variable')
                    self.absolute_effect_size    = meta_data.get('absolute_effect_size')
                    self.sample_size             = meta_data.get('sample_size')
                    try:
                        self.covariates          = meta_data.get('covariates')
                    except:
                        print("The dictionary of accompanying covariates was empty.")
                    self.data_file_name          = meta_data.get('data_file_name')
                    self.data_file_location      = meta_data.get('data_file_location')
                    self.meta_data_file_name     = meta_data.get('meta_data_file_name')
                    self.meta_data_file_location = meta_data.get('meta_data_file_location')
                    self.noise_loc               = meta_data.get('noise_loc')
                    self.noise_scale             = meta_data.get('noise_scale')
                    self.stats                   = meta_data.get('stats')
                    self.data                    = df    
                    self.rsquared                = meta_data.get('rsquared')
                    self.rsquared_adj            = meta_data.get('rsquared_adj')
                    
                    print("Successfully read in the meta-data specified at:" +
                          "\n     {}.".format(meta_data_path))
                       
                except:
                    print("Failed to read in the meta-data specified at:\n     {}.".format(meta_data_path)) 
                
            except:
                print("Failed to read in the .csv file specified at {}.".format(data_path)) 
        else:
            print("Invalid arguments inputted.")
            print("Enter the name and cardinality of the dependent variable, the absolute effect size,")
            print("the number of observations to be created, and a dictionary of covariates, or specify")
            print("the path to the appropriate data (.csv file) and meta-data (.txt file).")

      

    
    
    
    
    
    