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
from skgstat_uncertainty.components.utils import variomodel_to_dict, PERFORMANCE_MEASURES
from skgstat_uncertainty.components.plotting import metric_parcats


def measure_plot(models: List[VarioModel], container=st):
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
    measure = opts.selectbox('Evaluation measure', options=measures)
    
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

    # handle figure download
    do_download = fig_area.button('DOWNLOAD', key='download_measure_plot')
    if do_download:
        download_link = components.figure_download_link(fig)
        fig_area.write(download_link, unsafe_allow_html=True)


def metric_plot(models: List[VarioModel], metric_select: bool = True, container=st) -> None:
    # load the available metrics
    available_metrics = []
    for m in models:
        for metric in m.parameters.get('measures', {}).keys():
            if metric not in available_metrics and metric in PERFORMANCE_MEASURES:
                available_metrics.append(metric)
    
    # check if the user is allowed to select the used metrics
    metrics = container.multiselect('Use metrics', options=available_metrics, format_func=lambda k: PERFORMANCE_MEASURES.get(k))

    if len(metrics) == 0:
        container.info('At least one metric needs to be selected')
        return
        
    fig = metric_parcats(models=models, metrics=metrics)
    container.plotly_chart(fig, use_container_width=True)


def feature_enabler(container=st.sidebar):
    # set features on/off by default
    if 'show_metrics' not in st.session_state:
        st.session_state.show_metrics = True
    if 'show_params' not in st.session_state:
        st.session_state.show_params = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # create the controls
    container.checkbox('Show metrics plot', key='show_metrics', help="Parallel coordinate plot to rank model parameterizations by multiple metrics at once")
    container.checkbox('Show parameter interactions', key='show_params', help="Scatter plots of variogram parameter and model metrics to explore interactions")
    container.checkbox('enable filter & results', key='show_results', help='Filter parameterizations and create multiple result plots. Can be computationally expansive, so disable while still filtering')


def result_grid():
    pass

def main_app(api: API) -> None:
    st.title("Model comparison")
    st.markdown("In this chapter you can use different metric to assess model quality to come up with an automated selection process.")

    # add the selection box for the different compartments
    features_expander = st.sidebar.expander('SUB APPLICATIONS', expanded=True)
    feature_enabler(container=features_expander)

    # load the dataset, and interval to be used
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)
    models = interval.models

    # show the table
    table_expander = st.expander('MODEL DETAILS', expanded=True)
    table_anchor = components.model_table(models=models, variant='dataframe', container=table_expander)

    # metrics area
    if st.session_state.show_metrics:
        metrics_expander = st.expander('MODEL METRICS', expanded=True)
        metric_plot(models=models, container=metrics_expander)
            
    # add the Parameter interaction
    if st.session_state.show_params:
        measure_expander = st.expander('MEASURE DETAILS', expanded=True)
        measure_plot(models=models, container=measure_expander)
    
    # filter and results
    if st.session_state.show_results:
        # TODO: This is maybe a component?
        # start building the selection options get the measure
        measure_opts = st.sidebar.expander('SELECTION ALGORITHM', expanded=True)    
        measure = measure_opts.selectbox('Evaluation measure', options=list(PERFORMANCE_MEASURES.keys()), format_func=lambda k: PERFORMANCE_MEASURES.get(k))
        
        # Model selection section - get the models
        excluded_models = components.model_selection(measure,models=models, container=measure_opts)

        result_container = st.expander('RESULTS', expanded=True)
        user_exclueded_ids = result_container.multiselect(
            'EXCLUDED MODELS',
            options=[model.id for model in models],
            default=excluded_models,
            format_func=lambda k: f'ID #{k}'
        )

        # re-render the table
        components.model_table(models=models, variant='dataframe', excluded_models=user_exclueded_ids, table_anchor=table_anchor   )

        with st.spinner('Calculating...'):
            # sidebar
            comparison_expander = st.sidebar.expander('RESULT COMPARISON', expanded=True)
            
            num_plots = int(comparison_expander.number_input('Number of Plots', min_value=1, max_value=6, value=1))

            # build the columns
            if num_plots > 1 and num_plots < 5:
                cols = cycle(result_container.columns(2))
            elif num_plots >= 5:
                cols = cycle(result_container.columns(3))
            else:
                cols = cycle(result_container.columns(1))
            
            # get the fields
            kriging_fields = interval.kriging_fields

            for i in range(num_plots):
                container = next(cols)
                components.single_result_plot(
                    kriging_fields=kriging_fields,
                    excluded_models=user_exclueded_ids,
                    container=container,
                    key=i,
                    disable_download=False,
                )


if __name__ == '__main__':
    st.set_page_config(page_title='Model comparison', layout='wide')
    def run(data_path=None, db_name=None):
        api = API(data_path=data_path, db_name=db_name)
        main_app(api=api)
    
    import fire
    fire.Fire(run)
