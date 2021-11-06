from typing import List
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from itertools import cycle

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import VarioModel
from skgstat_uncertainty import components
from skgstat_uncertainty.components.utils import variomodel_to_dict


def measure_plot(models: List[VarioModel], container=st, option_container=st.sidebar) -> str:
    # get the models as list
    data = variomodel_to_dict(models, add_measures=True)

    # get the column layout
    opts, fig_area = container.columns((3, 6))

    opts.markdown("Use the controls to inspect the model parameters. The plot should help you set up a good selection algorithm in the sidebar.")

    # extract the measures
    measures = []
    for model in models:
        for measure in model.parameters.get('measures', {}).keys():
            if measure not in measures:
                measures.append(measure)

    # build the options
    measure = option_container.selectbox('Evaluation measure', options=measures)
    
    # build plot options
    chart_type = opts.selectbox('Plot type', ['box', 'scatter'])

    # build the plot
    fig = go.Figure()
    
    # switch option
    if chart_type == 'box':
        h = opts.checkbox('horizontal layout', value=False)
        # prepare the data
        grp = defaultdict(lambda: list())
        
        # build the data model
        for model in models:
            grp[model.parameters['model_params']['model']].append(model.parameters['measures'].get(measure))

        # add trace for every model
        for model_name, model_data in grp.items():
            fig.add_trace(go.Box(**{'x' if h else 'y': model_data, 'name': model_name.capitalize(), 'boxpoints': 'all'}))
        
        fig.update_layout(
            legend=dict(orientation='h', y=1.05, yanchor='bottom'),
            ** {
                'xaxis' if h else 'yaxis': dict(title=measure.upper()),
                'yaxis' if h else 'xaxis': dict(title='Model')
            }
        )
        
    elif chart_type == 'scatter':
        all_df = pd.DataFrame(data)
        # add controls
        ctl = opts.selectbox('Model parameter', [o for o in all_df.columns if o not in ['id', 'model', *measures]])
        use_grp = opts.checkbox('Segement data by model type', value=True)
        i = opts.checkbox('invert axis')
        
        # group by 
        if use_grp:
            grp = all_df.groupby('model')
        else:
            grp = (('all models', all_df),)

        for model_name, df in grp:
            fig.add_trace(
                go.Scatter(**{
                    'y' if i else 'x': df[ctl],
                    'x' if i else 'y': df[measure],
                    'mode': 'markers',
                    'marker': dict(size=8),
                    'name': model_name.capitalize()
                })
            )
        
        fig.update_layout(
            legend=dict(orientation='h', y=1.05, yanchor='bottom'),
            **{
                'yaxis' if i else 'xaxis': dict(title=ctl.upper()),
                'xaxis' if i else 'yaxis': dict(title=measure.upper())
            }
        )

    # plot and return the measure
    fig_area.plotly_chart(fig, use_container_width=True)
    
    return measure


def model_selection(measure: str, models: List[VarioModel], container = st.sidebar) -> List[int]:
    exculded_models = []

    # get the measures
    measures = [model.parameters['measures'][measure] for model in models]
    # create method selection
    SELECTION_TYPE = {
        'threshold': 'Discard models by threshold'
    }
    method = container.selectbox('Selection method', options=list(SELECTION_TYPE.keys()), format_func=lambda k: SELECTION_TYPE.get(k))


    # switch methods
    if method == 'threshold':
        threshold = container.selectbox('threshold', ['std', 'min'])
        multiplier = container.number_input(
            f'{threshold} multiplier', 
            value=1.0 if threshold == 'std' else 1.6, 
            min_value=0.0, 
            help=f"How many times the {threshold} is still considered to be valid?"
        )

        # calculate the actual threshold
        if threshold == 'std':
            thres = np.min(measures) + np.std(measures) * multiplier
        elif threshold == 'min':
            thres = np.min(measures) * multiplier
        
        # filter
        exculded_models = [model.id for model in models if model.parameters['measures'][measure] > thres]

        if len(exculded_models) == 0:
            container.warning('Currently, no models are excluded.')
        else:
            container.info(f'The current filter excludes {len(exculded_models)} of {len(models)} models')
        
    return exculded_models


def main_app(api: API) -> None:
    st.title("Model comparison")
    st.markdown("In this chapter you can use different metric to assess model quality to come up with an automated selection process.")

    # load the dataset, and interval to be used
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)
    models = interval.models

    # show the table
    table_anchor = components.model_table(models=models, variant='dataframe')

    # add the Parameter inspection
    measure_expander = st.expander('MEASURE DETAILS', expanded=True)
    measure_expander.markdown('## Parameter inspection')
    measure_opts = st.sidebar.expander('SELECTION ALGORITHM', expanded=True)
    measure = measure_plot(models=models, container=measure_expander, option_container=measure_opts)

    load_compare = st.checkbox('Done selection algorithm, calculate results.')

    # Model selection section
    excluded_models = model_selection(measure,models=models, container=measure_opts)

    if not load_compare:
        components.model_table(models=models, variant='dataframe', excluded_models=excluded_models, table_anchor=table_anchor)
        st.stop()
    # otherwiese contine

    # Analysis part
    st.markdown("## Comparison")
    st.markdown("You can find the models excluded by your selection algorithm below. Via the select control, it's possible to dismiss even more models or add some excluded bach.")

    user_exclueded_ids = st.multiselect(
        'EXCLUDED MODELS',
        options=[model.id for model in models],
        default=excluded_models,
        format_func=lambda k: f'ID #{k}'
    )

    # re-render the table
    components.model_table(models=models, variant='dataframe', excluded_models=user_exclueded_ids, table_anchor=table_anchor   )

    st.markdown("### Result Plots")
    with st.spinner('Calculating...'):
        # sidebar
        comparison_expander = st.sidebar.expander('RESULT COMPARISON', expanded=True)
        
        num_plots = comparison_expander.number_input('Number of Plots', min_value=1, max_value=6, value=1)

        # build the columns
        if num_plots > 1 and num_plots < 5:
            cols = cycle(st.columns(2))
        elif num_plots >= 5:
            cols = cycle(st.columns(3))
        else:
            cols = cycle(st.columns(1))
        
        # get the fields
        kriging_fields = interval.kriging_fields

        for i in range(num_plots):
            container = next(cols)
            components.single_result_plot(kriging_fields=kriging_fields, excluded_models=user_exclueded_ids, container=container, key=i, disable_download=False)

if __name__ == '__main__':
    api = API()
    main_app(api=api)
