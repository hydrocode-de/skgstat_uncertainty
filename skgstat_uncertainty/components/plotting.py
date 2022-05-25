from typing import List
from random import choice
from string import ascii_letters
import base64
from itertools import cycle
from collections import defaultdict

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skinfo.metrics import entropy

from skgstat_uncertainty.models import VarioParams, VarioConfInterval, VarioModel, VarioModelResult, DataUpload
from skgstat_uncertainty.components.utils import PERFORMANCE_MEASURES


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
        TARGET['entropy_field'] = 'Kriging estimation entropy map'
        TARGET['entropy_model'] = 'Model entropy contribution'
    
        # add model plots
        TARGET['model'] = 'Single model function'
        TARGET['models'] = 'Plot all models'

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
    
    # UNCERTAINT
    elif target.startswith('multi_'):
        mode, ident = target.split('_')
        cm = header[1].selectbox('Colorscale', options=CS, index=15 if mode=='multi' else 2, key=f'colorselect_{key}')

        # stack the stuff together
        fields = np.stack([res.content[ident] for res in kriging_fields if res.model.id not in excluded_models], axis=2)

        # calcualte the bounds width
        if mode == 'multi':
            _result = np.max(fields, axis=2) - np.min(fields, axis=2)

        # build the figure
        fig.add_trace(go.Heatmap(z=_result, colorscale=cm))
        fig.update_layout(
            title=TARGET.get(target),
            yaxis=dict(scaleanchor='x')
        )

    # ENTROPY MAPS
    elif target.startswith('entropy_'):
        mode, ident = target.split('_')
        cm = header[1].selectbox('Colorscale', options=CS, index=15 if mode=='multi' else 2, key=f'colorselect_{key}')
        # stack the stuff together
        fields = np.stack([res.content['field'] for res in kriging_fields if res.model.id not in excluded_models], axis=2)

        # get the confidence interval and experimental variogram
        conf_interval: VarioConfInterval = kriging_fields[0].model.confidence_interval
        obs = conf_interval.variogram.variogram.values

        # calculute the bins 
        bins = np.linspace(np.min(obs), np.max(obs), len(kriging_fields) - len(excluded_models))
        all_h = np.apply_along_axis(entropy, 2, fields, bins=bins, normalize=True)

        if ident == 'field':
            # use the all_h result
            _result = all_h
        elif ident == 'model':
            # calculate the h of the field
            MODS = {res.model.id: f'{res.model.model_type.capitalize()} model <ID={res.model.id}>' for res in kriging_fields}
            model_id = header[1].selectbox('Model result', options=list(MODS.keys()), format_func=lambda k: MODS.get(k), key=f'model{key}')
            
            # recreate the selected ids
            selected_ids = [res.model.id for res in kriging_fields if res.model.id not in excluded_models]
            _filtered = fields[:,:, [i for i, mod_id in enumerate(selected_ids) if mod_id != model_id]]
            field_result = np.apply_along_axis(entropy, 2, _filtered, bins=bins, normalize=True)
            _result = (all_h - field_result) / all_h 

        # build the figure
        fig.add_trace(go.Heatmap(z=_result, colorscale=cm))
        fig.update_layout(
            title=f'Entropy of all fields' if ident == 'field' else f'Entropy of {MODS[model_id]}',
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
    
    # ALL MODELS
    elif target == 'models':
        # check if excluded models should be displayed
        show_excluded = header[1].checkbox('Show excluded models', value=False)
        
        # load all models
        if show_excluded:
            MODS = [res.model for res in kriging_fields]
        else:
            MODS = [res.model for res in kriging_fields if res.model.id not in excluded_models]

        # load the conf interval of one model
        interval = MODS[0].confidence_interval

        # load the base variogram
        vario = interval.variogram

        # create the the plotting data
        x = np.linspace(0, vario.variogram.bins[-1], 100)

        # create the base graph
        fig = base_conf_graph(vario=vario, interval=interval, fig=fig)

        # add all models
        for model in MODS:
            # load the model
            V = model.variogram

            # add the trace
            fig.add_trace(go.Scatter(x=x, y=V.fitted_model(x), mode='lines', line=dict(color='green', width=0.5), name=f"{model.model_type.capitalize()} model <ID={model.id}>"))


    # change the size for just any figure
    fig.update_layout(
        height=600,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    # render figure
    container.plotly_chart(fig, use_container_width=True)

    if not disable_download:
        do_download = container.button('DOWNLOAD', key=f'download_{key}')
        if do_download:
            container.write(figure_download_link(fig), unsafe_allow_html=True)


def figure_download_link(figure: go.Figure, filename: str = None, template: str = 'plotly_white') -> str:

    # check if a filename was given
    if filename is None:
        filename = ''.join([choice(ascii_letters) for _ in range(16)])
    if not filename.endswith('.pdf'):
        filename += '.pdf'

    if template is not None:
        figure.update_layout(
            template=template
        )
    
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


def metric_parcats(models: List[VarioModel], metrics: List[str] = ['rmse', 'cv'], colors: List[str] = 'all', colorscale=None, percentiles: List[int] = [25, 50, 75], fig = None, col: int = 1, row: int = 1) -> go.Figure:
    """
    Parallel category plot for parameterization performance measures.
    For any given list of parameterized VarioModels, a ranking for every given
    metric is created. Each metric is repesented by a cateogry dimension in the plot.
    The ranking is then categoriezed into percentiles (4 quartiles by default) and the 
    parameterizations are ordered into dimension values to show patterns.
    The first dimension is always the model type and the parameterizations will be colored
    accordingly.
    """
    # make the figure
    if fig is None:
        fig = make_subplots(1, 1, specs=[[{'type': 'domain'}]])
    
    # make an index over unique model types
    model_idx = {t: i for i, t in enumerate(set([m.model_type for m in models]))}

    # build a colorarray
    if colorscale is None:
        if colors == 'all':
            colors = cycle(['#1f77b4', '#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        else:
            colors = cycle(colors)

        # build the colorscale from the colors
        colorscale = [[i / (len(model_idx) - 1), next(colors)] for t, i in model_idx.items()]
    else:
        colors = None

    # build the dimensions data structure
    dimensions = defaultdict(lambda: dict(values=[]))
    model_names = []

    # add each model
    for m in models:
        for metric in metrics:
            dimensions[metric]['values'].append(m.parameters['measures'].get(metric, np.nan))
        model_names.append(model_idx[m.model_type])

    # the first category dimension is just the model names
    cat_dimensions = [
        dict(label='Model type', values=model_names, categoryarray=list(model_idx.values()), ticktext=[k.capitalize() for k in model_idx.keys()])
    ]

    # build the ticktext label:
    p = percentiles
    ticktext = [f'<{p[0]}%', *[f'{p[i - 1]} - %{p[i]}%' for i in range(1, len(p))], f'>{p[-1]}%']
    
    # add each metric as a new dimension
    for metric, data in dimensions.items():
        # get the percentiles for this measure
        p = np.nanpercentile(data['values'], percentiles)
        
        # replace each with the highest fitting category, or 0 if None fits
        f = lambda x: ([0] + [i + 1 for i, b in enumerate(p) if x >= b]).pop()
        v = [f(_) for _ in data['values']]

        # order only ticks that are actually present
        arr = sorted(np.unique(v).tolist()) 

        # add dimension
        cat_dimensions.append(dict(label=PERFORMANCE_MEASURES.get(metric), values=v, ticktext=ticktext, categoryarray=arr))

    # build the figure
    fig.add_trace(go.Parcats(
        line=dict(color=model_names, colorscale=colorscale, shape='hspline'),
        dimensions=cat_dimensions,
        hoveron='color'
    ), row=row, col=col)
    
    return fig
