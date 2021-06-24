import os
import streamlit as st
import skgstat as skg
import numpy as np
import pandas as pd
import pickle
import glob
import json
import hashlib
from skgstat.plotting import backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# build the page fast to become responsive
"""
# Compare Models

This short application let's you compare different models
that were fitted with the previous application. With this
application, you can compare the different models, create or
update their interpolation and decide which one to use.

"""
st.sidebar.title('Parameters')

# build the sigma-level dropdown
simga_levels = {
    0: 'Precise observations',
    2: '2/256 observation uncertainty',
    5: '5/256 observation uncertainty',
    10: '10/256 observation uncertainty',
    15: '15/256 observation uncertainty',
    25: '25/256 observation uncertainty',
}
sigma = st.sidebar.selectbox(
    'Observation uncertainty',
    options=[0, 2, 5, 10, 15, 25],
    format_func=lambda k: simga_levels[k],
    index=2
)

# path
PATH = os.path.abspath(os.path.dirname(__file__))

# use skgstat plotly backend
backend('plotly')

# gobally build the Variogram
# get the data
coordinates, values = skg.data.pancake(N=150, seed=42).get('sample')

# estimate the variogram
vario = skg.Variogram(coordinates, values, n_lags=20)
vario.maxlag = 500


# define some functions to handle data
def load_error_bounds(sigma):
    # load the full MC simulation
    fname = os.path.join(PATH, '..', 'data', f'MC_results_50000_{sigma}.pickle')
    with open(fname, 'rb') as f:
        _d = pickle.load(f)
        ebins = _d['ebins']
    
    # claculate error margins
    error_bounds = np.column_stack((
        np.min(ebins, axis=1),
        np.max(ebins, axis=1)
    ))

    return error_bounds


def load_all_models(sigma) -> dict:
    models_list = list()

    # load all
    for fname in glob.glob(os.path.join(PATH, '../data/model_fits/*.json')):
        with open(fname, 'r') as f:
            params = json.load(f)

        # add
        models_list.extend([p for p  in params if p['sigma_obs'] == sigma])
        

    # add a numeric id
    for i, mod in enumerate(models_list):
        mod['id'] = i

    return models_list


def apply_model(params):
    x = np.linspace(0, vario.maxlag, 100)

    # get the model
    model_name = params.get('model')
    model = getattr(skg.models, model_name)

    # build the params
    args = [params.get('effective_range'), params.get('sill')]
    if model_name in ('matern', 'stable'):
        args.append(params.get('shape'))
    args.append(params.get('nugget'))

    # apply the model
    y = model(x, *args)

    return x, y


def filter_models(models, sigma, include_fit=True, fit_level=100, std_level=1):
    # select the correct ones
    filt = [p for p in models if p['sigma_obs'] == sigma]

    # get the std
    std = np.std([p['rmse'] for p in filt]) * std_level
    mean = np.mean([p['rmse'] for p in filt])

    # filter
    filtered = [p for p in filt if np.abs(p['rmse'] - mean) <= std or include_fit and p['fit'] >= fit_level]

    return filtered


def hash_parameters(params):
    # stringify the params
    string = json.dumps(params)

    # create the checksum
    md5 = hashlib.md5(string.encode()).hexdigest()

    return md5


def apply_kriging(params: dict):
    # clone the variogram above
    v = vario.clone()

    # set the model
    v.model = params.get('model')

    # build the argument array
    args = dict(
        fit_range=params.get('effective_range'),
        fit_sill=params.get('sill'),
        fit_nugget=params.get('nugget') 
    )
    if params.get('model') in ('matern', 'stable'):
        args['fit_shape'] = params.get('shape')
    
    # update the kwargs
    v._kwargs.update(args)

    # switch to manual fit
    v.fit_method = 'manual'

    # apply kriging
    x = y = range(500)
    krige = v.to_gs_krige()
    field, _ = krige.structured((x, y))

    return field


def save_kriging_cache(path, store):
    with open(path, 'wb') as f:
        pickle.dump(store, f)


# load all models and render a Table
models_list = load_all_models(sigma)
all_models = pd.DataFrame(models_list)
all_models.set_index('id', inplace=True)

st.sidebar.markdown('## Filter models')
include_fits = st.sidebar.checkbox(
    'Include models within Margin?',
    value=True
)

std_level = st.sidebar.number_input(
    'Include models within times standard deviation',
    min_value=0.5,
    max_value=10.,
    step=0.5,
    value=1.5
)

# apply the pre-filter
pre_filter = filter_models(models_list, sigma, include_fits, std_level=std_level)
pre_filter_ids = [p['id'] for p in pre_filter]
#filtered_models = 

excluded_models = st.sidebar.multiselect(
    'Manually exclude specific models from being run',
    options=[int(_) for _ in all_models.index],
    default=[p['id'] for p in models_list if p['id'] not in pre_filter_ids]

)

# Create a dataframe of filtered models
filtered_models_list = [p for p in models_list if p['id'] not in excluded_models]
# filtered_models = all_models.loc[~all_models.index.isin(excluded_models)]
filtered_models = pd.DataFrame(filtered_models_list)

r"""
## Filter Models
At $\frac{%d}{256}$ level uncertainty used, currently %d / %d models 
are selected for further analysis
""" % (sigma, len(filtered_models), len(all_models))
st.table(filtered_models)

# Kriging part
st.sidebar.markdown('## Kriging')
f"""
## Kriging

The {len(filtered_models)} models listed above are now used to interpolate
the observations using each of the selected models for Kriging. 

The application caches kriging results, but you can force a re-calculation.
As kriging can take some time, you need to activate that step manually.
"""
force_kriging = st.sidebar.checkbox(
    'Force re-calculation interpolations',
    value = False
)

run_kriging = st.sidebar.checkbox(
    'Activate kriging',
    value=False
)



# data handling
models_pickle_path = os.path.join(PATH, '../data/model_propagation_fields.pickle')
if os.path.exists(models_pickle_path) and not force_kriging:
    with open(models_pickle_path, 'rb') as f:
        MODEL_STORE = pickle.load(f)
else:
    MODEL_STORE = dict()

# check how much is needed
required_hashes = [hash_parameters(p) for p in filtered_models_list]
cached_models = [h for h in MODEL_STORE.keys()]
is_cached = [h in cached_models for h in required_hashes]

# Load anything that is already there
kriging_fields = {h: MODEL_STORE.get(h) for h, can_load in zip(required_hashes, is_cached) if can_load}

# Run if necessary
if run_kriging and sum(is_cached) < len(required_hashes):
    with st.spinner('Running kriging...'):
        progress_bar = st.progress(0)

        # get the model_ids
        filtered_models_ids = [int(_) for _ in filtered_models.index]
        
        for mod_id in filtered_models_ids:
            # get the params
            params = [p for p in models_list if p['id'] == mod_id][0]

            # check for existance
            param_hash = hash_parameters(params)
            if param_hash in MODEL_STORE.keys():
                # load from cache
                kriging_fields[mod_id] = MODEL_STORE.get(param_hash)
            else:
                # this action is costy
                field = apply_kriging(params)

                # add to cache
                MODEL_STORE[param_hash] = field
                kriging_fields[mod_id] = field

                # save cache
                save_kriging_cache(models_pickle_path, MODEL_STORE)

            # update the bar
            progress_bar.progress(int((mod_id + 1) / len(filtered_models_ids) * 100))

    # done
    st.success('Kriging completed.')
elif sum(is_cached) >= len(required_hashes):
    st.success(f'''
    All {len(required_hashes)} selected models found in the cache. 
    ''')
else:
    st.info(f'''
    Kriging not activated.
    
    Currently, {len(required_hashes)} interpolations are needed from which
    {sum(is_cached)} are found in the cache. The remaining {len(required_hashes) - sum(is_cached)}
    runs would need approx. {(len(required_hashes) - sum(is_cached)) * 16} seconds.
    
    Found a total amount of {len(cached_models)} fields in the cache.
    ''')

# Use the kriged fields
f"""
### Single fields

You can inspect the {sum(is_cached)} fields below and analyze their impact on the overall uncertainty.
"""
# controls for the quartils
(lo, hi) = st.sidebar.slider(
    'Select confidence interval',
    min_value=0,
    max_value=100,
    value=(10, 90),
    step=5
)

# calculate the uncertainty
if len(kriging_fields.keys()) == 0:
    st.warning('Currently no kriging fields are found.')
else:
    all_used_fields = np.stack(list(kriging_fields.values()), axis=2)

    lower, higher = (np.percentile(all_used_fields, lo, axis=2), np.percentile(all_used_fields, hi, axis=2))

    # create the figure
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=higher - lower, colorscale='Hot')
    )
    fig.update_layout(
        title='Value range of confidence interval',
        yaxis=dict(scaleanchor='x', )
    )
    st.plotly_chart(fig, use_container_width=True)

    # get the original pancake
    pan = skg.data.pancake_field().get('sample')

    # inspect individual fields
    single_fields = []
    for params, h, is_loaded in zip(filtered_models_list, required_hashes, is_cached):
        # append a dictionary about the single fields
        field = MODEL_STORE.get(h)
        d = {
            'id': params.get('id'),
            'model': params.get('model').capitalize(),
            'model fit': '%.1f %%' % params.get('fit'),
            'model fit RMSE': '%.1f' % params.get('rmse'),
            'in invterval': '%.1f %%' % (np.sum((field >= lower) & (field <= higher).astype(int)) / (np.multiply(*field.shape)) * 100),
            'value range': '[%d, %d]' % (int(np.min(field)), int(np.max(field))),
            'field RMSE': '%.1f' % (np.sqrt(np.sum(np.power(field - pan, 2))))
        }

        single_fields.append(d)

    # build general overview
    st.text(f'''
    Interpolation value range:   [{int(np.min(all_used_fields))}, {int(np.max(all_used_fields))}]
    Confidence interval width:   [{int(np.min(higher - lower))}, {int(np.max(higher - lower))}]
    ''')

    # build the dataframe
    single_info = pd.DataFrame(single_fields)
    single_info.set_index('id', inplace=True)
    st.table(single_info)


# show the applied models
"""
### Fitted models

To inspect the models more in depth, you can find the original error bounds 
of the experimental variogram. The models in use can be found in the right panel,
the excluded models are shown for reference in the left panel.

"""
error_bounds = load_error_bounds(sigma)

# apply all models
included, excluded = [], []
for param in models_list:
    # apply the model within maxlag
    x, y = apply_model(param)

    # store the name
    name = f"{param.get('model').capitalize()} #{param.get('id')}"
    
    # check if this is an excluded model
    if param.get('id') in excluded_models:
        excluded.append((x, y, name))
    else:
        included.append((x, y, name))

# add the figure
fig = make_subplots(1, 2, shared_yaxes=True)

for col, mods in zip((1, 2), (excluded, included)):
    # plot the excluded models
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,0], mode='lines', line=dict(color='gray'), fill=None, name='lower bound'),
        row=1, col=col
    )
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,1], mode='lines', line=dict(color='gray'), fill='tonexty', name='upper bound'),
        row=1, col=col
    )

    # go for the models
    for x, y, name in mods:
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name=name),
            row=1, col=col
        )

st.plotly_chart(fig, height='500px', use_container_width=True)
