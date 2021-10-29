from typing import List

from skgstat_uncertainty.models import VarioModel


def variomodel_to_dict(models: List[VarioModel]) -> List[dict]:
    # build up the data for the table
    data = list()

    for model in models:
        d = {'id': model.id}

        # get the parameters
        d.update(model.parameters.get('model_params', {}))

        # append
        data.append(d)

    return data
