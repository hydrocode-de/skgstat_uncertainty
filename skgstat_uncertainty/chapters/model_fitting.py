from typing import Tuple, List
import warnings

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skgstat import models
from scipy import stats

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import VarioModel, VarioParams, VarioConfInterval
from skgstat_uncertainty.processor import fit
from skgstat_uncertainty import components


MODELS = {
    'spherical': 'Spherical',
    'exponential': 'Exponential',
    'gaussian': 'Gaussian',
    'cubic': 'Cubic',
    'stable': 'Stable',
    'matern': 'Matérn'
}


def base_graph(vario: VarioParams, interval: VarioConfInterval) -> go.Figure:
    # load the interval
    bounds = interval.spec['interval']
    quartiles = interval.spec.get('quartiles', ['?', '?'])

    # load the bins
    x = vario.variogram.bins

    # create the figure
    fig = go.Figure()

    # create the plot
    fig.add_trace(
        go.Scatter(x=x, y=[b[0] for b in bounds], mode='lines', line_color='grey', fill=None, name=f'{quartiles[0]}% - percentile')
    )
    fig.add_trace(
        go.Scatter(x=x, y=[b[1] for b in bounds], mode='lines', line_color='grey', fill='tonexty', name=f'{quartiles[1]}% - percentile')
    )
    fig.update_layout(
        legend=dict(orientation='h'),
        xaxis=dict(title='Lag', showgrid=False),
        yaxis=dict(title=f"{vario.variogram.estimator.__name__.capitalize()} semi-variance", showgrid=False),
    )

    return fig


def apply_model(vario: VarioParams, interval: VarioConfInterval, figure: go.Figure, other_models: List[VarioModel] = []) -> Tuple[go.Figure, dict]:
    # create the controls in the sidebar
    st.sidebar.title('Fitting Parameters')

    # get the variogram and bounds
    variogram = vario.variogram
    bounds = interval.spec['interval']
    max_bound = float(np.nanmax([b[1] for b in bounds]).round(2))
    max_bin = float(np.nanmax(variogram.bins).round(2))

    model = st.sidebar.selectbox('Theoretical model', options=list(MODELS.keys()), format_func=lambda k: MODELS.get(k))
    nugget, sill = st.sidebar.slider(
        'Nugget and sill',
        min_value=0.0,
        max_value=float(np.round(1.3 * max_bound, 2)),
        value=[0.0, float(np.round(0.98 * max_bound, 2))]
    )
    _range = st.sidebar.slider(
        'Effective Range',
        min_value=0.0,
        max_value=max_bin,
        value=float(np.round(0.5*max_bin, 2))
    )
    if model == 'stable':
        _s = st.sidebar.slider('Shape Parameter', min_value=0.05, max_value=2.0, value=0.5)
    elif model == 'matern':
        _s = st.sidebar.slider('Smoothness Parameter', min_value=0.2, max_value=10.0, value=2.0)
    else:
        _s = None

    # apply the model
    variogram.model = model
    
    # apply the parameters
    x = np.linspace(0, max_bin, 100)
    func = getattr(models, model)
    if _s is None:
        y = [func(h, _range, sill, nugget) for h in x]
    else:
        y = [func(h, _range, sill, _s, nugget) for h in x]

    # if there are other models, plot them
    if len(other_models) > 0:
        for mod in other_models:
            figure.add_trace(
                go.Scatter(x=x, y=mod.apply_model(x=x), mode='lines', line_color='grey', line_width=0.8)
            )
    
    # add the trace for the new model
    figure.add_trace(
        go.Scatter(x=x, y=y, mode='lines', line_color='green', line_width=3, name=f"{model.capitalize()} model")
    )

    # current parameter dict
    params = {
        'model': model,
        'range': _range,
        'sill': sill,
        'nugget': nugget,
        'shape': _s
    }
    # return the changed figure back
    return figure, params


def evaluate_fit(vario: VarioParams, interval: List[Tuple[float, float]], params: dict, other_models: List[VarioModel] = []) -> Tuple[float, float]:
    # calcualte the rmse
    rmse = fit.rmse(vario, interval, params)
    
    v = vario.variogram
    cv = fit.cv(vario, params)

    # create four cols
    first, left, center, right = st.columns(4)

    # check if there are other models
    if len(other_models) > 0:
        # get population statistics
        pop_rmse = np.nanmean([m.parameters['measures'].get('RMSE', np.nan) for m in other_models])
        pop_cv = np.nanmean([m.parameters['measures'].get('cross-validation', np.nan) for m in other_models])
        rank = stats.rankdata([cv, *[m.parameters['measures'].get('cross-validation', np.nan) for m in other_models]], method='min')[0]
        
        # fancy
        rank = '1st' if rank == 1 else rank
        rank = '2nd' if rank == 2 else rank
        rank = '3rd' if rank == 3 else rank
        rank = f'{rank}th' if not isinstance(rank, str) and rank > 3 else rank

        # calculate deviation
        rmse_dev = (rmse - pop_rmse).round(1)
        cv_dev = (cv - pop_cv).round(1)
        rank_dev = f"of {len(other_models) + 1} models"
    else:
        rmse_dev = None
        cv_dev = None
        rank = None
        rank_dev = None

    first.markdown('### \n### All models')
    left.metric("Model Rank", value=rank, delta=rank_dev, delta_color='off')
    center.metric("Fit - RMSE", value=rmse.round(1), delta=rmse_dev, delta_color='inverse')
    right.metric("Model - cross validation", value=cv.round(1), delta=cv_dev, delta_color='inverse')

    # check if the given model was already used
    if len(other_models) > 0:
        # get the warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            # RMSE
            try:
                mod_rmse_ = np.nanmean([m.parameters['measures'].get('RMSE', np.nan) for m in other_models if m.parameters['model_params']['model'] == params['model']])
                mod_rmse = (rmse - mod_rmse_).round(1)
            except Warning:
                mod_rmse = None
            
            # CROSS-VALIDATION
            try:
                mod_cv_ = np.nanmean([m.parameters['measures'].get('cross-validation', np.nan) for m in other_models if m.parameters['model_params']['model'] == params['model']])
                mod_cv = (cv - mod_cv_).round(1)
            except Warning:
                mod_cv = None
        
            # CV RANK
            try:
                mod_rank = stats.rankdata([cv, *[m.parameters['measures'].get('cross-validation', np.nan) for m in other_models if m.parameters['model_params']['model'] == params['model']]], method='min')[0]
        
                # fancy
                mod_rank = '1st' if mod_rank == 1 else mod_rank
                mod_rank = '2nd' if mod_rank == 2 else mod_rank
                mod_rank = '3rd' if mod_rank == 3 else mod_rank
                mod_rank = f'{mod_rank}th' if not isinstance(mod_rank, str) and mod_rank > 3 else mod_rank
                mod_rank_dev = f"of {len([1 for m in other_models if m.parameters['model_params']['model'] == params['model']]) + 1} {params['model']} models"
            except Warning:
                mod_rank_dev = None    
        
    else:
        mod_rmse = None
        mod_cv = None
        mod_rank = None
        mod_rank_dev = None
    
    first.markdown(f"# \n### {params['model'].capitalize()} models only")
    left.metric(f"{params['model'].capitalize()} model rank", value=mod_rank, delta=mod_rank_dev, delta_color='off')
    center.metric("Fit - RMSE", value=rmse.round(1), delta=mod_rmse, delta_color='inverse')
    right.metric("Model - cross validation", value=cv.round(1), delta=mod_cv, delta_color='inverse')


    # show the current params
    # params.update({'RMSE': rmse, 'cross-validation': cv})
    # left.markdown('## Current parameters')
    # left.json(params)
    
    return rmse, cv


def save_handler(api: API, interval: VarioConfInterval, params: dict, fit_measures: dict) -> VarioModel:
    # create the save object
    parameters = dict(model_params=params, measures=fit_measures)

    # save
    model = api.set_vario_model(interval.id, params['model'], **parameters)

    return model


def main_app(api: API) -> None:
    st.title('Variogram Model Fitting')
    st.markdown("")
    save_btn = st.empty()

    # create the dataset creator  
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)

    # load existing models
    load_other = st.sidebar.checkbox(
        'Load other models', 
        value=False, 
        help=f"Load already fitted models for {interval.name} for comparison. Slows the application down."
    )
    if load_other:
        prior_models = api.filter_vario_model(conf_id=interval.id)
        if len(prior_models) > 0:
            st.info(f"{len(prior_models)} other models are found for the confidence interval {interval.name}")
    else:
        prior_models = []

    # create the base figure and show
    fig = base_graph(vario=vario, interval=interval)

    # if still active apply fitting
    fig, params = apply_model(vario=vario, interval=interval, figure=fig, other_models=prior_models)
    st.plotly_chart(fig, use_container_width=True)

    # create the evaluation figures
    rmse, cv = evaluate_fit(vario=vario, interval=interval, params=params, other_models=prior_models)

    # handle save
    st.markdown('##\n')
    st.write("""<hr style="margin: "0 2rem" />""", unsafe_allow_html=True)
    do_save = st.button('SAVE', key='save_model')
    do_save2 = st.sidebar.button('SAVE', key='save_model2')
    
    if do_save or do_save2:
        model = save_handler(api=api, params=params, interval=interval, fit_measures={'RMSE': rmse, 'cross-validation': cv})
        st.success(f"Saved {model.model_type} model with ID {model.id}")
    else:
        st.stop()
    
    # finished, so stop
    st.success('Chapter finished. Continue with the next one.')
    st.stop()


if __name__ == '__main__':
    api = API()
    main_app(api=api)
