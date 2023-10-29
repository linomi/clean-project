from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(cache= True,manifest_file='../brain_data/brain_observatory_manifest.json')
depth = boc.get_all_imaging_depths()
all_areas = boc.get_all_targeted_structures()


boc.get_cell_specimens()