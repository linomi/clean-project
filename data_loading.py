from filter_cells import filter
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(cache= True,manifest_file='../brain_data/brain_observatory_manifest.json')
import numpy as np
def load_data(experiment_id):
    nwb = boc.get_ophys_experiment_data(ophys_experiment_id=experiment_id)
    num_cells = len(nwb.get_cell_specimen_ids())
    nwb.get_cell_specimen_indices()
    targeted_cells = filter(nwb= nwb,train_reliability=0.5,test_reliability=0.3)
    train_movie = nwb.get_stimulus_template('natural_movie_three')
    train_movie = np.expand_dims(train_movie,axis = 3)
    dff = nwb.get_dff_traces(cell_specimen_ids=nwb.get_cell_specimen_ids())
    train_trace = dff[1][:,np.array(nwb.get_stimulus_table('natural_movie_three'))[:,2]]
    train_trace = train_trace.reshape((num_cells,10,3600))
    train_trace = np.mean(train_trace,axis = 1)
    train_trace = (train_trace - np.expand_dims(train_trace.min(axis = 1),axis = 1))/(np.expand_dims(train_trace.max(axis = 1),axis = 1)- np.expand_dims(train_trace.min(axis = 1),axis = 1))
    train_trace = train_trace.transpose()
    train_trace = train_trace[:,targeted_cells]




    val_movie = nwb.get_stimulus_template('natural_movie_one')
    val_movie = np.expand_dims(val_movie,axis = 3)
    dff = nwb.get_dff_traces(cell_specimen_ids=nwb.get_cell_specimen_ids())
    val_trace = dff[1][:,np.array(nwb.get_stimulus_table('natural_movie_one'))[:,2]]
    val_trace = val_trace.reshape((num_cells,10,900))
    val_trace = np.mean(val_trace,axis = 1)
    val_trace = (val_trace - np.expand_dims(val_trace.min(axis = 1),axis = 1))/(np.expand_dims(val_trace.max(axis = 1),axis = 1)- np.expand_dims(val_trace.min(axis = 1),axis = 1))
    val_trace = val_trace.transpose()
    val_trace = val_trace[:,targeted_cells]
    
<<<<<<< HEAD
    runing_speed = nwb.get_running_speed()
=======
    runing_speed = nwb.get_running_speed()[0][np.array(nwb.get_stimulus_table('natural_movie_three'))]
>>>>>>> e09922bd7580d5b2504c887de3ab5745d0bd7b67
    return (train_movie,train_trace),(val_movie,val_trace),runing_speed
