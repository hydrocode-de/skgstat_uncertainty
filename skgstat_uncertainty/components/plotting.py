from typing import List
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from random import choice
from string import ascii_letters
import base64
from skinfo.metrics import entropy

from skgstat_uncertainty.models import VarioParams, VarioConfInterval, VarioModel, VarioModelResult, DataUpload


def single_result_plot(kriging_fields: List[VarioModelResult], excluded_models: List[int] = [], container=st, key='', disable_download=True):
    # Targets
    TARGET = {
        'single_field': 'Single model kriging estimate',
        'single_sigma': 'Single model kriging error',
        'violin': 'Model violin plot'
    }
    
    # add field specific plotting options
    if len(kriging_fields) - len(excluded_models) >= 2:
        # add multi plots
        TARGET['multi_field'] = 'Kriging estimation uncertainty bounds'
        TARGET['multi_sigma'] = 'Kriging error uncertainty bounds'
    
        # add model plots
        TARGET['model'] = 'Single model function'

    # Colorscales
    CS = ['Blackbody', 'Bluered', 'Blues','Cividis', 'Earth', 'Electric', 'Greens', 'Greys', 'Hot', 'Jet', 'Picnic','Portl', 'Rainbow', 'RdBu', 'Reds', 'Viridis', 'YlGnBu', 'YlOrRd']

    # use a header
    # header = container.columns(2)
    header = [container, container]
    # select target
    target = header[0].selectbox('Plot type', options=list(TARGET.keys()), format_func=lambda k: TARGET.get(k), key=f'target{key}')

    # build the figure
    fig = go.Figure()
    
    # SINGLE
    if target.startswith('single_'):
        # Select a result set
        MODS = {res.model.id: f'{res.model.model_type.capitalize()} model <ID={res.model.id}>' for res in kriging_fields}
        mod_id = header[1].selectbox('Model result', options=list(MODS.keys()), format_func=lambda k: MODS.get(k), key=f'model{key}')
        result = [res for res in kriging_fields if res.model.id==mod_id].pop()

        # get the key
        ident = target.split('_')[1]

        # build the figure
        fig.add_trace(go.Heatmap(z=result.content[ident]))
        fig.update_layout(yaxis=dict(scaleanchor='x'))
    
    # VIOLON
    elif target == 'violin':
        header[1].info('Large (slow) plot')

        # gather all the needed data
        for res in kriging_fields:
            # if it is excluded, continue
            if res.model.id in excluded_models:
                continue
            
            # get the data
            data = np.array(res.content['field']).flatten()
            
            # add a violin
            fig.add_trace(go.Violin(x=data, name=f"{res.model.model_type.capitalize()} model <ID={res.model.id}>"))
        fig.update_layout(legend=dict(orientation='h'))
    
    # UNCERTAINTY AND ENTROPY
    elif target.startswith('multi_') or target.startswith('entropy_'):
        mode, ident = target.split('_')
        cm = header[1].selectbox('Colorscale', options=CS, index=15 if mode=='multi' else 2, key=f'colorselect_{key}')

        # stack the stuff together
        fields = np.stack([res.content[ident] for res in kriging_fields if res.model.id not in excluded_models], axis=2)

        # calcualte the bounds width
        if mode == 'multi':
            _result = np.max(fields, axis=2) - np.min(fields, axis=2)
        else:
            # calculate the bins by Scott's rule
            bins = np.histogram_bin_edges(fields.flatten(), bins='scott')
            _result = np.apply_along_axis(entropy, 2, fields, bins=bins, normalize=True)

        # build the figure
        fig.add_trace(go.Heatmap(z=_result, colorscale=cm))
        fig.update_layout(
            title=TARGET.get(target),
            yaxis=dict(scaleanchor='x')
        )

    # SINGLE MODEL
    elif target == 'model':
        MODS = {res.model.id: f'{res.model.model_type.capitalize()} model <ID={res.model.id}>' for res in kriging_fields}
        mod_id = header[1].selectbox('Model result', options=list(MODS.keys()), format_func=lambda k: MODS.get(k), key=f'model{key}')
        
        # load the data to display
        model: VarioModel = [res.model for res in kriging_fields if res.model.id==mod_id].pop()

        # load conf interval
        interval = model.confidence_interval

        # load vario
        vario = model.confidence_interval.variogram

        # create the base graph
        fig = base_conf_graph(vario=vario, interval=interval, fig=fig)

        # build the model
        V = model.variogram
        x = np.linspace(0, V.bins[-1], 100)
        y = V.fitted_model(x)

        # add the trace
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='green', width=3), name=f"{model.model_type.capitalize()} model <ID={model.id}>"))
        fig.update_layout(
            legend=dict(orientation='h')
        )

    # change the size for just any figure
    fig.update_layout(height=600)

    # render figure
    container.plotly_chart(fig, use_container_width=True)

    if not disable_download:
        do_download = container.button('DOWNLOAD', key=f'download_{key}')
        if do_download:
            container.write(figure_download_link(fig), unsafe_allow_html=True)


def figure_download_link(figure: go.Figure, filename: str = None) -> str:

    # check if a filename was given
    if filename is None:
        filename = ''.join([choice(ascii_letters) for _ in range(16)])
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # create a byte string
    img_bytes = figure.to_image(format='pdf')
    
    # base64 encode the image
    b64 = base64.b64encode(img_bytes)

    # create the anchor tag
    return f"""<a href="data:application/pdf;base64,{b64.decode()}" download="{filename}">Download {filename}</a>"""


def dataset_plot(dataset: DataUpload, disable_download=True, key='', container=st) -> None:
    # switch the figure type
    if dataset.data_type == 'field' or dataset.data_type == 'auxiliary':
        fig = go.Figure(
            go.Heatmap(z=dataset.data['field'])
        )

        fig.update_layout(
            height=750,
            yaxis=dict(scaleanchor='x'), 
            plot_bgcolor='rgba(0,0,0,0)'
        )

        container.plotly_chart(fig, use_container_width=True)
    elif dataset.data_type == 'sample':
        fig = go.Figure(
            go.Scatter(x=dataset.data['x'], y=dataset.data['y'], mode='markers', marker=dict(size=8, symbol='cross', color=dataset.data['v']))
        )

        fig.update_layout(
            height=750,
            yaxis=dict(scaleanchor='x'), 
            plot_bgcolor='rgba(0,0,0,0)'
        )

        container.plotly_chart(fig, use_container_width=True)
    else:
        container.json(dataset.data)
        return
    
    # if this is run, an actual plot was created
    if not disable_download:
        do_download = container.button('DOWNLOAD', key=f'download_{key}')
        if do_download:
            container.write(figure_download_link(fig), unsafe_allow_html=True)


def base_conf_graph(vario: VarioParams, interval: VarioConfInterval, fig: go.Figure = None) -> go.Figure:
    # load the interval
    bounds = interval.spec['interval']
    quartiles = interval.spec.get('quartiles', ['?', '?'])

    # load the bins
    x = vario.variogram.bins

    # create the figure
    if fig is None:
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
