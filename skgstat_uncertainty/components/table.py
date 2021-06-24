import streamlit as st
import pandas as pd


def table_export_options(df: pd.DataFrame, export_formats=['LaTeX', 'CSV', 'JSON', 'Markdown', 'HTML'], container=None, key=None):
    # if no container set, create one
    if container is None:
        container = st
    
    # build the columns
    if len(export_formats) > 1:
        options, table_area  = container.beta_columns((1, 9))
    elif len(export_formats) == 1:
        options = table_area = container
    else:
        raise AttributeError('At least one export format needs to be set')
    
    # check options
    if len(export_formats) == 1:
        fmt = export_formats[0]
    else:
        fmt = options.radio('Format', options=export_formats, key=key)
    
    # handle export
    if fmt.lower() == 'latex':
        table_area.code(df.to_latex(index=None))
    elif fmt.lower() == 'csv':
        table_area.code(df.to_csv(index=None))
    elif fmt.lower() == 'json':
        table_area.code(df.to_json(orient='records', indent=4))
    elif fmt.lower() == 'markdown':
        table_area.code(df.to_markdown(None))
    elif fmt.lower() == 'html':
        table_area.code(df.to_html(index=None))