from typing import List

from skgstat_uncertainty.models import VarioModel


def variomodel_to_dict(models: List[VarioModel], add_measures = False) -> List[dict]:
    # build up the data for the table
    data = list()

    for model in models:
        d = {'id': model.id}

        # get the parameters
        d.update(model.parameters.get('model_params', {}))

        if add_measures:
            d.update(model.parameters.get('measures', {}))
        # append
        data.append(d)

    return data
