"""
Utility function for handling toolbox data in the scope of a 
streamlit application.

The only function so far can consume 'field' and 'sample' DataUpload from
the database and create a plotly figure of the data. This figure can be 
returned or exported into a png, which is base64 encoded and injected
into a data-url. This is saved to the DataUpload as 'thumbnail' and can
be loaded in streamlit as a preview of the dataset.
"""
from typing import Union
from io import BytesIO
import base64

import plotly.graph_objects as go


def create_thumbnail(data: 'DataUpload', return_type: str = 'base64', **kwargs) -> Union[str, go.Figure]:
    fig = kwargs.get('fig')

    if fig is None:
        fig = go.Figure()
        # check if this is a field or sample
        if data.data_type == 'field':
            fig.add_trace(go.Heatmap(z=data.data['field'], colorscale=kwargs.get('colorscale', 'Earth_r'), showscale=False))
            fig.update_layout(yaxis=dict(scaleanchor='x'))
        elif data.data_type == 'sample':
            fig.add_trace(go.Scatter(
                x=data.data['x'],
                y=data.data['y'],
                mode='markers',
                marker=dict(size=9, color=data.data['v'], colorscale=kwargs.get('colorscale', 'Electric'), 
                showscale=False,
                )
            ))
        else:
            fig.add_annotation(text='No Preview', x=0.5, y=0.5, yref="paper", xref="paper", showarrow=False, font=dict(size=22, color="grey"))

    # set some general layout
    fig.update_layout(
        height=kwargs.get('height', 200),
        width=kwargs.get('width', 200),
        yaxis=dict(visible=False),
        xaxis=dict(visible=False),
        margin=dict(t=0, b=0, r=0, l=0)
    )
    
    if return_type.lower() in ('fig', 'figure'):
        return fig
    elif return_type.lower() in ('b64', 'base64'):
        b = BytesIO()
        fig.write_image(b, format='png')
        b.seek(0)

        return f"data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"
