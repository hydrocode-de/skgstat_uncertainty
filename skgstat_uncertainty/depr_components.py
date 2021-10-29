from typing import Union, List
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from skgstat_uncertainty.models import DataUpload, VarioModelResult


def multi_plot_field_heatmaps(fields: Union[List[DataUpload], List[VarioModelResult]], container = st) -> None:
    # determine the number of cols needed
    if len(fields) > 6:
        cols = 3
    elif len(fields) > 2:
        cols = 2
    else: 
        cols = 1
    
    # get the number of rows
    rows = np.ceil(len(fields) / cols).astype(int)

    # create the cols
    columns = container.columns(cols)

    i = 0
    for r in range(rows):
        for c in range(cols):
            # get the data
            data: dict = fields[i].content if hasattr(fields[i], 'content') else fields[i].data
            field = data['field'] if 'field' in data else data['kriging_field']
            
            # create the figure
            fig = go.Figure(go.Heatmap(z=field))
            fig.update_layout(yaxis=dict(scaleanchor='x'), plot_bgcolor='rgba(0,0,0,0)')

            # add
            columns[c].plotly_chart(fig, use_container_width=True)


