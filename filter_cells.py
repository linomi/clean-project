from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(cache= True,manifest_file='../brain_data/brain_observatory_manifest.json')
import json 
filter_json = """
[
    {
        "field": "reliability_nm1_a",
        "op": "between",
        "value": [
            0.5,
            1
        ]
    },
    {
        "field": "reliability_nm2",
        "op": "between",
        "value": [
            0.5,
            1
        ]
    }
]
"""

filters = json.loads(filter_json)
def filter(nwb,train_reliability = 0.7,test_reliability=0.3):
    filters[0]['value'] = [test_reliability,1]
    filters[1]['value'] = [train_reliability,1]
    cells = nwb.get_cell_specimen_ids()
    filtered_cells = boc.get_cell_specimens(filters=filters,ids= cells)
    targeted_cells_ids = [cell['cell_specimen_id'] for cell in filtered_cells]
    targeted_cells = nwb.get_cell_specimen_indices(cell_specimen_ids = targeted_cells_ids)
    return targeted_cells
