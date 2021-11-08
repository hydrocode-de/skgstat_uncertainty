from typing import List
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from random import choice
from string import ascii_letters
import base64

from skgstat_uncertainty.models import VarioModelResult, DataUpload


def single_result_plot(kriging_fields: List[VarioModelResult], excluded_models: List[int] = [], container=st, key='', disable_download=True):
    # Targets
    TARGET = {
        'single_field': 'Single model kriging estimate',
        'single_sigma': 'Single model kriging error',
        'violin': 'Model violin plot'
    }
    if len(kriging_fields) - len(excluded_models) >= 2:
        TARGET['multi_field'] = 'Kriging estimation uncertainty bounds'
        TARGET['multi_sigma'] = 'Kriging error uncertainty bounds'
    
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
    
    # UNCERTAINTY
    elif target.startswith('multi_'):
        cm = header[1].selectbox('Colorscale', options=CS, index=15, key=f'colorselect_{key}')
        ident = target.split('_')[1]

        # stack the stuff together
        fields = np.stack([res.content[ident] for res in kriging_fields if res.model.id not in excluded_models], axis=2)

        # calcualte the bounds width
        bwidth = np.max(fields, axis=2) - np.min(fields, axis=2)

        # build the figure
        fig.add_trace(go.Heatmap(z=bwidth, colorscale=cm))
        fig.update_layout(
            title=TARGET.get(target),
            yaxis=dict(scaleanchor='x')
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


def dataset_plot(dataset: DataUpload, container=st) -> None:
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
