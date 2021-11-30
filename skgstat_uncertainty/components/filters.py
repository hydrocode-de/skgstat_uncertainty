from typing import List
import streamlit as st
import numpy as np

from skgstat_uncertainty.models import VarioModel

def model_selection(measure: str, models: List[VarioModel], container = st.sidebar) -> List[int]:
    exculded_models = []

    # get the measures
    measures = [model.parameters['measures'][measure] for model in models]
    
    # create method selection
    SELECTION_TYPE = {
        'threshold': 'Discard models by threshold',
        'take_amount': 'Take N models',
    }
    method = container.selectbox('Selection method', options=list(SELECTION_TYPE.keys()), format_func=lambda k: SELECTION_TYPE.get(k))

    # switch methods
    if method == 'threshold':
        threshold = container.selectbox('threshold', ['std', 'min'])
        multiplier = container.number_input(
            f'{threshold} multiplier', 
            value=1.0 if threshold == 'std' else 1.6, 
            min_value=0.0 if threshold == 'std' else 1.0, 
            help=f"How many times the {threshold} is still considered to be valid?"
        )

        # calculate the actual threshold
        if threshold == 'std':
            thres = np.nanmin(measures) + np.nanstd(measures) * multiplier
        elif threshold == 'min':
            thres = np.nanmin(measures) + np.abs(np.nanmin(measures) * (multiplier - 1.0))
        
        container.write(f"T: {thres}")
        
        # filter
        exculded_models = [model.id for model in models if model.parameters['measures'][measure] > thres]

        if len(exculded_models) == 0:
            container.warning('Currently, no models are excluded.')
        else:
            container.info(f'The current filter excludes {len(exculded_models)} of {len(models)} models')

    # take N models
    elif method == 'take_amount':
        # get the amount
        distribute_models = container.checkbox('Take the same amount of every model type', value=False)
        startval = int(0.5 * len(models)) + 1 if not distribute_models else 2
        n_models = int(container.number_input('Take N models', value=startval, min_value=1, max_value=len(models)))
        
        if distribute_models:
            # get the model types
            model_types = set([model.parameters['model_params']['model'] for model in models])
            # take the models
            exculded_models = []
            for model_type in model_types:
                # get the models of current type
                _models = [model for model in models if model.parameters['model_params']['model'] == model_type]
                
                # argsort the measures
                idx = np.argsort([model.parameters['measures'][measure] for model in _models])
                
                # get the models ids
                model_ids = np.array([model.id for model in _models])
                
                # take the models
                exculded_models.extend(model_ids[idx][n_models:].tolist())
        else:
            # make an index over all models
            idx = np.argsort(measures)
            exculded_models = np.array([model.id for model in models])[idx][n_models:].tolist()
        
    return exculded_models
