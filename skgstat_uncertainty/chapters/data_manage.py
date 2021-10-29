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
    'sample': 'Sample an existing dataset'
}
def upload_handler(api: API) -> DataUpload:
    st.markdown("""Upload data to the application. The following MIME types are supported:
    
    \r* _csv_: A csv file containing the headers 'x' and 'y' for coordinates and 'v' for values
    \r* _asc_, _txt_: a space delimeted file of a 2D field (rows x cols)

    """)    

    # create the upload handler    
    uploaded_file = st.file_uploader('Choose the data', ['csv', 'asc', 'txt'])

    # if no file uploaded stop the application
    if uploaded_file is None:
        st.stop()
    
    # get the mime type
    data_name, mime = os.path.splitext(uploaded_file.name)

    if mime == '.csv':
        data = pd.read_csv(uploaded_file)
                
        if not 'x' in data.columns and 'y' in data.columns and 'v' in data.columns:
            st.error('CSV files need to specify the columns x and y for coordinates and v for values.')
            st.stop()

        # save data
        dataset = api.set_upload_data(
            data_name,
            'sample',
            x=data.x.values.tolist(),
            y=data.y.values.tolist(),
            v=data.z.values.tolist()
        )
    elif mime == '.asc' or mime == '.txt':
        data = np.loadtxt(uploaded_file)

        # save the data
        dataset = api.set_upload_data(
            data_name,
            'field',
            field=data.tolist()
        )
    else:
        st.error(f'File of type {mime} not supported.')
        st.stop()

    return dataset


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


def main_app(api: API):
    st.title('Data Upload manager')
    st.markdown("Use this chapter to upload new datasets and create new samples.")
    

    # check what the user wants to do
    action = st.radio('Specify action', options=list(ACT.keys()), format_func=lambda k: ACT.get(k))

    if action == 'upload':
        # upload handler
        dataset = upload_handler(api=api)

        # add auxiliary upload
        components.upload_auxiliary_data(dataset=dataset, api=api)

        # plot the uploaded data
        fig = go.Figure(go.Heatmap(z=dataset.data['field']))
        fig.update_layout(
            height=750,
            yaxis=dict(scaleanchor='x'), 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


    elif action == 'sample':
        # create the data select
        dataset = components.data_selector(api=api, stop_with='data', data_type='field', container=st.sidebar)

        # dev only 
        sample_dense_data(dataset=dataset, api=api)

    st.success('Finished!')




if __name__=='__main__':
    def run(db_name='data.db'):
        api = API(db_name=db_name)
        main_app(api=api)

    import fire
    fire.Fire(run)
    