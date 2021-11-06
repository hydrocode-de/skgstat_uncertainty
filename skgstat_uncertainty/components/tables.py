from typing import List
import streamlit as st
import pandas as pd

from skgstat_uncertainty.models import VarioModel
from .utils import variomodel_to_dict


def model_table(models: List[VarioModel], variant='table', excluded_models=[], container=st, table_anchor=None):
    # get the models as list
    data = variomodel_to_dict(models, add_measures=True)

    if table_anchor is None:
        table_anchor = container.empty()

    if variant == 'dataframe':
        df = pd.DataFrame(data)
        styled_df = df.style.apply(lambda r: ['background-color: %s' % ('salmon' if  r['id'] in excluded_models else '')] * len(r.values), axis=1)
        table_anchor.dataframe(styled_df)
    else:
        table_anchor.table(data)
    
    return table_anchor
