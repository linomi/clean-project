from filter_cells import filter
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(cache= True,manifest_file='../brain_data/brain_observatory_manifest.json')
import numpy as np
def load_data(experiment_id,switch_data = False,train_reliablity = 0.5,test_reliablity = 0.6):
    nwb = boc.get_ophys_experiment_data(ophys_experiment_id=experiment_id)
    num_cells = len(nwb.get_cell_specimen_ids())
    targeted_cells = filter(nwb= nwb,train_reliability=train_reliablity,test_reliability=test_reliablity)
    if len(targeted_cells) == 0: 
        raise Exception('two tight reliablity, no cells founded by forced criteria ')
    
    train_movie = nwb.get_stimulus_template('natural_movie_three')
    train_movie = np.expand_dims(train_movie,axis = 3)
    dff = nwb.get_dff_traces(cell_specimen_ids=nwb.get_cell_specimen_ids())
    train_trace = dff[1][:,np.array(nwb.get_stimulus_table('natural_movie_three'))[:,2]]
    train_trace = train_trace.reshape((num_cells,10,3600))
    train_trace = np.mean(train_trace,axis = 1)
    train_trace = (train_trace - np.expand_dims(train_trace.min(axis = 1),axis = 1))/(np.expand_dims(train_trace.max(axis = 1),axis = 1)- np.expand_dims(train_trace.min(axis = 1),axis = 1))
    train_trace = train_trace.transpose()
    train_trace = train_trace[:,targeted_cells]
    running_speed_train = nwb.get_running_speed()[0][np.array(nwb.get_stimulus_table('natural_movie_three'))[:,2]]
    running_speed_train = running_speed_train.reshape((10,-1)).mean(axis =0)
    running_speed_train[np.where(running_speed_train=='Nan')] = 0
   




    val_movie = nwb.get_stimulus_template('natural_movie_one')
    val_movie = np.expand_dims(val_movie,axis = 3)
    dff = nwb.get_dff_traces(cell_specimen_ids=nwb.get_cell_specimen_ids())
    val_trace = dff[1][:,np.array(nwb.get_stimulus_table('natural_movie_one'))[:,2]]
    val_trace = val_trace.reshape((num_cells,10,900))
    val_trace = np.mean(val_trace,axis = 1)
    val_trace = (val_trace - np.expand_dims(val_trace.min(axis = 1),axis = 1))/(np.expand_dims(val_trace.max(axis = 1),axis = 1)- np.expand_dims(val_trace.min(axis = 1),axis = 1))
    val_trace = val_trace.transpose()
    val_trace = val_trace[:,targeted_cells]
    running_speed_val = nwb.get_running_speed()[0][np.array(nwb.get_stimulus_table('natural_movie_one'))[:,2]]
    running_speed_val = running_speed_val.reshape((10,-1)).mean(axis = 0)

    running_speed_val[np.where(running_speed_train=='Nan')] = 0    

    
    if switch_data: 
        return (val_movie,val_trace,running_speed_val),(train_movie,train_trace,running_speed_train)
    else:
        return (train_movie,train_trace,running_speed_train),(val_movie,val_trace,running_speed_val)
