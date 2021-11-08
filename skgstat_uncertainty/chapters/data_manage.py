import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty import components
from skgstat_uncertainty.processor import sampling


ACT = {
    'upload': 'Uplaod new dataset',
    'sample': 'Sample an existing dataset',
    'list': 'List existing dataset'
}


def sample_dense_data(dataset: DataUpload, api: API):
    # create the sidebar
    st.sidebar.title('Parameters')

    # sampling settings
    sampling_container = st.sidebar.expander('SAMPLING SETTINGS', expanded=True)
    sampling_strategy = sampling_container.selectbox(
        'Sampling Strategy',
        ['random', 'regular grid']
    )

    # switch the sampling method
    if sampling_strategy == 'random':
        # add the controls
        N = sampling_container.number_input('Sample size', value=150, min_value=10)
        seed = sampling_container.number_input('Seed', value=42, help="Seed the random selection to make your results repeatable")

        # apply random sample
        coords, values = sampling.random(dataset.data['field'], N=int(N), seed=int(seed))
    
    elif sampling_strategy == 'regular grid':
        # get the grid
        grid = np.asarray(dataset.data['field'])
        # add the controls
        gridder = sampling_container.selectbox('Specify grid', options=['shape', 'N of points', 'spacing'], help="Specify how the regular grid should be created")
        if gridder == 'N of points':
            N = sampling_container.number_input('Sample size', value=64, help="The grid might not match the sample size exactly. Only if N**1/2 == int")
            args = dict(N=N)
        elif gridder == 'shape':
            sampling_container.write('Specify the target shape like: 32x32')
            shape = sampling_container.text_input('Grid shape', value="8x8", help="You have to separate the shape for each dimension by a 'x'")
            args = dict(shape=[int(_) for _ in shape.split('x')])
        elif gridder == 'spacing':
            sampling_container.write('Specify the spacing for each axis like: 5,5')
            spac = sampling_container.text_input('Grid spacing', value='5,5', help="You have to separate the spacing for each dimension by a comma.")
            args = dict(spacing=[int(_) for _ in spac.split(',')])

        # add offset
        offset = sampling_container.number_input('Offset', value=0, min_value=0, max_value=int(np.max(grid.shape) / 2), step=1, help="Offset the grid from the outer field border")
        args['offset'] = offset
        try:
            coords, values = sampling.grid(grid, **args)
        except Exception as e:
            st.error("Oops. That did not work. Probably the grid setting make no sense for the dataset?")
            st.stop()

    # show data as table
    with_table = sampling_container.checkbox('Show data as table', value=False)

    # build the figure
    fig = go.Figure()
    data: list = dataset.data.get('field')
    # data.reverse()
    fig.add_trace(
        go.Heatmap(z=np.transpose(data))
    )
    fig.add_trace(
        go.Scatter(
            x=[c[0] for c in coords],
            y=[c[1] for c in coords],
            mode="markers",
            marker=dict(color='black', size=6, symbol='cross'),
            text=[f"{c}: {v}" for c, v in zip(coords, values)]
        )
    )
    fig.update_layout(
        height=750,
        yaxis=dict(scaleanchor='x'), 
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # check if we need a table as well
    if with_table:
        plot_area, table_area = st.columns((7,3))
        # table_area.table([{'x': c[0], 'y': c[1], 'value': v} for c, v in zip(coords, values)])
        table_area.dataframe([{'x': c[0], 'y': c[1], 'value': v} for c, v in zip(coords, values)], height=400)
    else:
        plot_area = st.empty()
    plot_area.plotly_chart(fig, use_container_width=True)

    st.info("""Check the sample shown above. If you are satisfied, choose a name for the new sample and hit save. 
    Afterwards, you can select the new dataset from the dropdown, or create another sample.""")

    dataset_name = st.text_input('Dataset Name')
    if dataset_name != "":
        do_save = st.button('SAVE', key='save1')
        do_save2 = sampling_container.button('SAVE', key='save2')

        # check if we need to save
        if do_save or do_save2:
            dataset = api.set_upload_data(
                dataset_name,
                'sample',
                field_id=dataset.id,
                x=[c[0] for c in coords],
                y=[c[1] for c in coords],
                v=values
            )
            st.success(f'Dataset {dataset_name} saved! Reload and choose it from the dropdown.')
            st.button('RELOAD')

    # as long as we are in this app, we need to stop
    st.stop()


def list_datasets(api: API, container=st):
    # select a dataset
    all_names = api.get_upload_names()
    data_id = container.selectbox('DATASET', options=list(all_names.keys()), format_func=lambda k: all_names.get(k))
    dataset = api.get_upload_data(id=data_id)

    # preview data
    container.title(f"{dataset.data_type.upper()} dataset")
    components.dataset_plot(dataset)

    # some basic stats
    stats = [
        {'Stat': 'Estimated experimental variograms', 'Value': len(dataset.variograms)},
        {'Stat': 'Total number of fitted models', 'Value': np.sum([[len(cv.models) for cv in v.conf_intervals] for v in dataset.variograms])}
    ]

    container.markdown('## Related data')
    container.table(stats)


def main_app(api: API):
    st.title('Data Upload manager')
    st.markdown("Use this chapter to upload new datasets and create new samples.")
    

    # check what the user wants to do
    action = st.radio('Specify action', options=list(ACT.keys()), format_func=lambda k: ACT.get(k))

    if action == 'upload':
        # upload handler
        dataset = components.upload_handler(api=api, can_select=False)

        # add auxiliary upload
        components.upload_auxiliary_data(dataset=dataset, api=api)
        st.markdown('### Preview')
        components.dataset_plot(dataset=dataset)
        
            # check if this dataset has origin information
        if 'origin' in dataset.data:
            origin = dataset.data['origin']

            oexp = st.expander('DATASET INFO', expanded=True)
            oexp.markdown(f'## Origin information\n{origin}')

    elif action == 'sample':
        # create the data select
        dataset = components.data_selector(api=api, stop_with='data', data_type='field', container=st.sidebar)

        # dev only 
        sample_dense_data(dataset=dataset, api=api)
    
    elif action == 'list':
        list_datasets(api=api)


if __name__=='__main__':
    def run(db_name='data.db'):
        api = API(db_name=db_name)
        main_app(api=api)

    import fire
    fire.Fire(run)
    