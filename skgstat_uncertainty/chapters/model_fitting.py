from typing import Dict, Tuple, List
import warnings
from collections import defaultdict

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skgstat import models
from scipy import stats

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import VarioModel, VarioParams, VarioConfInterval
from skgstat_uncertainty.processor import fit
from skgstat_uncertainty import components
from skgstat_uncertainty.components.utils import MODELS, PERFORMANCE_MEASURES


def apply_model(vario: VarioParams, interval: VarioConfInterval, figure: go.Figure, other_models: List[VarioModel] = [], expander_container=st.sidebar) -> Tuple[go.Figure, dict]:
    # get the variogram and bounds
    variogram = vario.variogram
    bounds = interval.spec['interval']
    max_bound = float(np.nanmax([b[1] for b in bounds]).round(2))
    max_bin = float(np.nanmax(variogram.bins).round(2))

    model = expander_container.selectbox('Theoretical model', options=list(MODELS.keys()), format_func=lambda k: MODELS.get(k))
    nugget, sill = expander_container.slider(
        'Nugget and sill',
        min_value=0.0,
        max_value=float(np.round(1.3 * max_bound, 2)),
        value=[0.0, float(np.round(0.98 * max_bound, 2))]
    )
    _range = expander_container.slider(
        'Effective Range',
        min_value=0.0,
        max_value=max_bin,
        value=float(np.round(0.5*max_bin, 2))
    )
    if model == 'stable':
        _s = expander_container.slider('Shape Parameter', min_value=0.05, max_value=2.0, value=0.5)
    elif model == 'matern':
        _s = expander_container.slider('Smoothness Parameter', min_value=0.2, max_value=10.0, value=2.0)
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


def parameterization_evaluation(vario: VarioParams, interval: List[Tuple[float, float]], params: dict, other_models: List[VarioModel] = []) -> Tuple[Dict[str, float], dict]:
    # create a dictionary to hold all measures
    measures = {}
    measure_params = defaultdict(lambda: {})

    # filter down if no other models are given
    no_mod = False
    if len(other_models) == 0:
        st.info(f"No other parameterization are found. This will deactivate the DIC and structural risk minimization measures.")
        no_mod = True
    if sum([m.model_type == params['model'] for m in other_models]) == 0:
        st.info(f"To calculate DIC or SRM, you should at least parameterize 5 {params['model']} models.")
        no_mod = True
    
    # filter
    MEAS = {k: v for k, v in PERFORMANCE_MEASURES.items() if k not in ['dic', 'srm'] or not no_mod}
    
    # ask the user for all performance measures to include
    metrics = st.multiselect('Select performance measures', options=list(MEAS.keys()), format_func=lambda k: MEAS.get(k), default=['rmse', 'cv'])

    # go for each metric
    for metric in metrics:
        # RMSE
        if metric == 'rmse':
            measures['rmse'] = fit.rmse(vario, interval, params)
        
        # Cross-validation
        elif metric == 'cv':
            cv_exp = st.sidebar.expander('Cross-validation')
            n = len(vario.variogram.values)
            measure_params['cv']['k'] = int(cv_exp.number_input('Sampling size', min_value=1, value=1, 
                max_value=int(0.25 *n), 
                help="Use 1 for leave-one-out cross-validation. If the number is increased, the robustness will be tested against larger sample sizes"
            ))
            measure_params['cv']['max_iter'] = int(cv_exp.number_input('Maximum iterations', min_value=1, value=n, max_value=n, 
                help="Decrease to speed up calculation. This will decrease the accuracy of the cross-validation"
            ))
            measures['cv'] = fit.cv(vario, params, k=measure_params['cv']['k'], max_iter=measure_params['cv']['max_iter'])
        
        # DIC
        elif metric == 'dic':
            dic_exp = st.sidebar.expander('DIC')
            measure_params['dic']['method'] = dic_exp.selectbox('DIC variation', options=['mean', 'median', 'mode', 'gelman'], format_func=lambda k: k.capitalize())
            vario_collection = [v for v in other_models if v.model_type == params['model']]
            measures['dic'] = fit.dic(vario, params, vario_collection, measure_params['dic']['method'])
        
        # Empirical Risk
        elif metric == 'er':
            er_exp = st.sidebar.expander('Empirical Risk')
            measure_params['er']['method'] = er_exp.selectbox('Loss function', options=['mae', 'binary'], format_func=lambda k: k.upper())
            measures['er'] = fit.empirical_risk(vario, interval, params, measure_params['er']['method'])
        
        # Structural risk minimization
        elif metric == 'srm':
            srm_exp = st.sidebar.expander('Structural Risk Minimization')
            measure_params['srm']['weight'] = srm_exp.slider('Conf. Interval <--> Complexity', min_value=0.0, max_value=2.0, value=1.0)
            measure_params['srm']['method'] = srm_exp.selectbox('Training Error method', options=['mae', 'binary'], format_func=lambda k: k.upper())
            measure_params['srm']['pD_method'] = srm_exp.selectbox('eff. Parameter method', options=['mean', 'median', 'mode', 'gelman'], format_func=lambda k: k.capitalize())


            vario_collection = [v for v in other_models if v.model_type == params['model']]
            measures['srm'] = fit.structural_risk(vario, interval, params, vario_collection, 
                weight=measure_params['srm']['weight'], 
                method=measure_params['srm']['method'],
                pD_method=measure_params['srm']['pD_method']
            )

    return measures, measure_params


def evaluation_metric_badges(measures: Dict[str, float], container=st) -> None:
    """
    Visualize the current measures using metric badges
    """
    # number of elements
    n = len(measures)

    # set up the columns
    cols = container.columns(n)

    # add all 
    for col, (metric, value) in zip(cols, measures.items()):
        col.metric(PERFORMANCE_MEASURES[metric], value=np.round(value, 1))


def save_handler(api: API, interval: VarioConfInterval, params: dict, measures: Dict[str, float], measure_params: Dict[str, Dict[str, float]]) -> VarioModel:
    # create the save object
    parameters = dict(model_params=params, measures=measures, measure_params=measure_params)

    # save
    model = api.set_vario_model(interval.id, params['model'], **parameters)

    # TODO: re-calculate everything related to DIC and SRM
    for name in ['srm', 'dic']:
        if name in measures:
            # get the other models of same type
            others = [m for m in interval.models if m.model_type == model.model_type]

            for m in others:
                params = m.parameters
                params['measures'][name] = measures[name]
                params.get('measure_params', {}).update({name: measure_params[name]})
                m.parameters = params
                m.save()

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
    fig = components.base_conf_graph(vario=vario, interval=interval)

    # if still active apply fitting
    fitting_expander = st.sidebar.expander('MODEL PARAMETERIZATION', expanded=True)
    fig, params = apply_model(vario=vario, interval=interval, figure=fig, other_models=prior_models, expander_container=fitting_expander)
    st.plotly_chart(fig, use_container_width=True)

    # create the evaluation figures
    measures, measure_params = parameterization_evaluation(vario=vario, interval=interval, params=params, other_models=prior_models)

    # show the metrics
    evaluation_metric_badges(measures=measures)

    # handle save
    st.markdown('##\n')
    st.write("""<hr style="margin: "0 2rem" />""", unsafe_allow_html=True)
    do_save = st.button('SAVE', key='save_model')
    do_save2 = st.sidebar.button('SAVE', key='save_model2')
    
    if do_save or do_save2:
        model = save_handler(api=api, params=params, interval=interval, measures=measures, measure_params=measure_params)
        st.success(f"Saved {model.model_type} model with ID {model.id}")
    else:
        st.stop()
    


if __name__ == '__main__':
    st.set_page_config(page_title='Model parameterization', layout='wide')
    def run(data_path=None, db_name=None):
        api = API(data_path=data_path, db_name=db_name)
        main_app(api=api)
    
    import fire
    fire.Fire(run)
