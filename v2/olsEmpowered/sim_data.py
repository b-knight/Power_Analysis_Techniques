import os
import ast
import time
import json
import random 
import numpy as np
import pandas as pd
from datetime import datetime   

class sim_data:

    # constructor
    def __init__(self, dv_name = None, dv_cardinality = None,
                 absolute_effect_size = None,
                 sample_size = None, covariates_dict = None,
                 noise_loc = None, noise_scale = None,
                 data_path = None, meta_data_path = None):
                      
        # create data generating method
        def create_dataframe(absolute_effect_size, covariates_dict, 
                             sample_size, noise_loc, noise_scale):
            data_dict = {}

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
        & (sample_size is not None) & (covariates_dict is not None)
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
                df = create_dataframe(absolute_effect_size, covariates_dict, sample_size, 
                                      noise_loc, noise_scale)
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

                # create log file
                items_to_log = {}
                items_to_log.update({'file_name': data_file_name})
                items_to_log.update({'dv_name': dv_name})
                items_to_log.update({'dv_cardinality': dv_cardinality})
                items_to_log.update({'treatment_variable': 'treated'})
                items_to_log.update({'absolute_effect_size': absolute_effect_size})
                items_to_log.update({'sample_size': sample_size})
                items_to_log.update({'covariates': covariates_dict})
                description = pd.DataFrame(df.describe()).to_dict()
                items_to_log.update({'noise_loc': noise_loc})
                items_to_log.update({'noise_scale': noise_scale})
                items_to_log.update({'stats': description})

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
                self.covariates              = covariates_dict
                self.data_file_name          = data_file_name
                self.data_file_location      = current_dir + '/data'
                self.meta_data_file_name     = data_file_name[0:-4] + "_log_file.txt"
                self.meta_data_file_location = current_dir + '/data/log_files'
                self.noise_loc               = noise_loc
                self.noise_scale             = noise_scale
                self.stats                   = description
                self.data                    = df
                
            except:
                print("Failed to generate the simulation data as specified.")
                
        # if a data_path is present, read data instead of creating it
        elif ((dv_name is None) & (dv_cardinality is None)
        & (sample_size is None) & (covariates_dict is None)
        & (absolute_effect_size is None)
        & (noise_loc is None) & (noise_scale is None)
        & (data_path != None) & (meta_data_path != None)):
            try:
                df = pd.read_csv(data_path) 
                print("Successfully read in the .csv file specified at:" + 
                      "\n     {}.".format(data_path))
                try:
                    meta_string = open(meta_data_path, 'r').read()
                    meta_data = ast.literal_eval(meta_string)
                    print("Successfully read in the meta-data specified at:" +
                          "\n     {}.".format(meta_data_path))
                    
                    # set class variables
                    self.dv_name                 = meta_data.get('dv_name')
                    self.dv_cardinality          = meta_data.get('dv_cardinality')
                    self.treatment_variable      = meta_data.get('treatment_variable')
                    self.absolute_effect_size    = meta_data.get('absolute_effect_size')
                    self.sample_size             = meta_data.get('sample_size')
                    self.covariates              = meta_data.get('covariates')
                    self.data_file_name          = meta_data.get('data_file_name')
                    self.data_file_location      = meta_data.get('data_file_location')
                    self.meta_data_file_name     = meta_data.get('meta_data_file_name')
                    self.meta_data_file_location = meta_data.get('meta_data_file_location')
                    self.noise_loc               = meta_data.get('noise_loc')
                    self.noise_scale             = meta_data.get('noise_scale')
                    self.stats                   = meta_data.get('stats')
                    self.data                    = df                    
                    
                except:
                    print("Failed to read in the meta-data specified at:\n     {}.".format(meta_data_path)) 
                
            except:
                print("Failed to read in the .csv file specified at {}.".format(data_path)) 
        else:
            print("Invalid arguments inputted.")
            print("Enter the name and cardinality of the dependent variable, the absolute effect size,")
            print("the number of observations to be created, and a dictionary of covariates, or specify")
            print("the path to the appropriate data (.csv file) and meta-data (.txt file).")
   