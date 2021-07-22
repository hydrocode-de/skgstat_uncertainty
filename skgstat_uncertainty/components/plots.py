from typing import Callable, List
import streamlit as st
import numpy as np
from skgstat import Variogram
import plotly.graph_objects as go


def detailed_kriged_plot(field_load_func: Callable[..., np.ndarray], params: List[dict], container=None, vario: Variogram = None):
    # get a streamlit instance if no container passed
    if container is None:
        container = st
    
    # define fixed layout
    layout = dict(
        yaxis=dict(scaleanchor='x'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )

    # get the number of fields
    n = len(params)
    if n == 0:
        return
    elif n == 1:
        rows = cols = 1
    else: 
        if n / 2 <= 3:
            cols = 2
        # elif n / 3 <= 3:
        #     cols = 3
        else:
            cols = 3
        rows = int(n / cols) + (n - cols * int(n / cols))
    
    # build the containers
    cells = [container.beta_columns(cols) for _ in range(rows)]

    # extract observations if variogram was given
    if vario is not None:
        obs = vario.coordinates
        obs_x = obs[:,0] - np.min(obs[:,0])
        obs_y = obs[:,1] - np.min(obs[:,1])
        vals = vario.values

    i = 0

    for row in cells:
        for col in row:
            if i == len(params):
                break
            # get the parameters and load the field
            p = params[i]
            field = field_load_func(p['model_md5'])

            fig = go.Figure(go.Heatmap(z=field.T, colorscale='Earth'))
            
            # add the observations
            if vario is not None:
                fig.add_trace(
                    go.Scatter(x=obs_x, y=obs_y, mode='markers', marker=dict(size=5, color='white'), text=[str(_) for _ in vals], name='Observations')
                )
            fig.update_layout(
                title=f"ID={p['id']} {p['model'].capitalize()} model based interpolation"
            )
            fig.update_layout(layout)

            col.plotly_chart(fig, use_container_width=True)
            i += 1
