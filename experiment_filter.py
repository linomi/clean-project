from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import json

boc = BrainObservatoryCache(manifest_file="../brain_data/brain_observatory_manifest.json")
depth = boc.get_all_imaging_depths()
all_areas = boc.get_all_targeted_structures()
def filter(area,minimum_reliablity):      
    filter_json = """
    [
        {
            "field": "reliability_nm3",
            "op": "between",
            "value": [
                0.5,
                1
            ]
        },
        {
            "field": "area",
            "op": "in",
            "value": [
                "VISl"
            ]
        }
    ]
    """
        
    filters = json.loads(filter_json)
    filters[1]['value'] = [area]
    filters[0]['value'] = [minimum_reliablity,1]
            
    cells = boc.get_cell_specimens(filters=filters)
    cells = [cell['cell_specimen_id'] for cell in cells]
    experiments = boc.get_ophys_experiments(targeted_structures=['VISl'],imaging_depths=depth,session_types=['three_session_A'],require_eye_tracking=True,cell_specimen_ids=cells)
    ids = [experiment['id'] for experiment in experiments ]
    return ids