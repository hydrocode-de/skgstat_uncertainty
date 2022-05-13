import streamlit as st
from typing import List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_card_select import card_select

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty import components
from skgstat_uncertainty.processor import sampling


ACT = {
    'upload': 'Uplaod new dataset',
    'sample': 'Sample an existing dataset',
    'list': 'List existing dataset',
    'edit': 'Edit existing dataset',
}

ADD_BTN = "data:image/png;base64,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"


#@st.experimental_memo
def _options_from_dataset_names(_api: API, datasets: dict, add_button: bool = True) -> List[dict]:
    if add_button:
        # create a container with an add button
        options = [dict(option='new', title='Create new dataset', image=ADD_BTN)]
    else:
        options = []
    
    # add each dataset as preview
    for data_id, name in datasets.items():
        d = dict(option=data_id, title=f"[ID: {data_id}] {name}")

        # get the dataset from db
        dataset = _api.get_upload_data(id=data_id)

        # handle description
        d['description'] = dataset.data['description'][:250] if 'description' in dataset.data else '<i>This dataset has no description</i>'

        if 'thumbnail' in dataset.data:
            d['image'] = dataset.data['thumbnail']
        
        # insert
        options.insert(0, d)

    # return
    return options


def dataset_grid(api: API) -> None:
    """
    Create a grid of all existing datasets. When clicked, the dataset is loaded
    for viewing. In viewing mode, the dataset can be edited or deleted. The grid
    does also include a button for creating new datasets
    """
    # get the upload names
    all_datasets = api.get_upload_names()

    # get the options
    options = _options_from_dataset_names(api, all_datasets)

    # check for data_id
    data_id = card_select(options=options, spacing=5, md=3, lg=3)

    # if None selected, stop
    if data_id is None:
        st.stop()
    elif data_id == 'new':
        # create new dataset
        st.session_state.action = 'new'
        st.experimental_rerun()
    else:
        st.session_state.action = 'view'
        st.session_state.data_id = data_id
        st.experimental_rerun()


def button_panel(can_resample: bool = False, container=st) -> None:
    # build the columns in the container
    cols = container.columns(4 if can_resample else 3)

    # add the buttons
    back = cols[0].button('BACK TO LIST')
    edit = cols[1].button('EDIT DATASET')
    delete = cols[-1].button('DELETE DATASET')
    
    if can_resample:
        resample = cols[2].button('RE-SAMPLE DATASET')
    else:
        resample = False

    # check the action
    if back:
        del st.session_state['data_id']
        del st.session_state['action']
        st.experimental_rerun()
    elif edit:
        st.session_state.action = 'edit'
        st.experimental_rerun()
    elif delete:
        st.session_state.action = 'delete'
        st.experimental_rerun()
    elif resample:
        st.session_state.action = 'sample'
        st.experimental_rerun()


def action_view(api: API) -> None:
    # get the dataset
    dataset = api.get_upload_data(id=st.session_state.data_id)
    data = dataset.data

    # build the page
    st.title(dataset.upload_name)
    st.info(f"This dataset is licensed under: _{components.utils.LICENSES.get(data.get('license', '__no__'), 'no license found')}_")

    # button list
    button_expander = st.expander('ACTIONS', expanded=True)
    button_panel(can_resample=dataset.data_type == 'field', container=button_expander)

    # main description
    left, right = st.columns((6, 4))
    left.markdown('### Description')
    right.markdown('### Origin')
    if 'description' in data:
        left.markdown(data['description'], unsafe_allow_html=True)
    else:
        left.info('Edit the dataset to add a description')
    if 'origin' in data:
        right.markdown(data['origin'], unsafe_allow_html=True)
    else:
        right.info('Edit the dataset to add the origin')

    st.markdown('### Preview')
    components.dataset_plot(dataset, disable_download=False)

    # debug area
    exp = st.expander('RAW database record')
    exp.json(dataset.to_dict())


def upload_view(api: API) -> None:
    # Title
    st.title('Upload a new dataset')
    st.info('As of now, the Dataset will be named exactly like the uploaded file. If you would like to change the name, you need to edit the dataset afterwards.')
    
    # upload handler
    dataset = components.upload_handler(api=api, can_select=False)

    # add the preview
    dataset.update_thumbnail()

    # add auxiliary data
    components.upload_auxiliary_data(dataset=dataset, api=api)

    # preview
    st.markdown('## Preview upload')
    components.dataset_plot(dataset, disable_download=True)

    st.markdown('## Finished?')
    go_back = st.button('Back')
    if go_back:
        st.session_state.action = 'list'
        st.experimental_rerun()


def edit_view(api: API) -> None:
    # get the dataset
    dataset = api.get_upload_data(id=st.session_state.data_id)

    # Title 
    st.title(f'Edit {dataset.upload_name}')

    # edit form
    edit_dataset(dataset=dataset, api=api)


def delete_view(api: API) -> None:
    # get the dataset
    dataset = api.get_upload_data(id=st.session_state.data_id)

    # Title
    st.title(f'Delete {dataset.upload_name}')
    st.error("**CAUTION** This action cannot be undone. If you delete the data, it will be permanently lost.")
     
    # check whatesle will be deleted
    varios = dataset.variograms
    cis = []
    for v in varios:
        cis.extend(v.conf_intervals)
    models = []
    for c in cis:
        models.extend(c.models)
    
    # inform on cascade deletes
    if len(varios) > 0:
        st.warning(f'The following data objects in the database depend on `{dataset.upload_name}` and will be **deleted as well**')

        st.table([{'Object type': n, 'Instances': len(m)} for n, m in zip(('Empirical variograms', 'Confidence Intervals', 'Variogram Models'), (varios, cis, models))])

    # acknowledge the deletion
    ack = st.checkbox('I understand the consequences of this action')
    if len(varios) > 0:
        ack2 = st.checkbox('I also understand that dependend data will be permanently deleted')
    else:
        ack2 = True

    # add cancel button
    st.markdown('Are you sure to delete?')
    l, r, _ = st.columns((2, 2, 4))
    cancel = l.button('CANCEL')
    if cancel:
        st.session_state.action = 'view'
        st.experimental_rerun()
    
    # Do the deletion if acknowledged
    if ack and ack2:
        delete = r.button('DELETE NOW')
    
        if delete:
            api.delete_upload_data(id=dataset.id)
            del st.session_state['data_id']
            del st.session_state['action']
            st.experimental_rerun()


def sample_view(api: API) -> None:
    # get the dataset
    dataset = api.get_upload_data(id=st.session_state.data_id)

    # Title
    st.title(f'Re-Sample {dataset.upload_name}')
    st.markdown('Use this little sub-app to create a new sample from the selected dense dataset or field. The new sample can be used just like any other dataset')

    sample_dense_data(dataset=dataset, api=api)


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

    with st.form('Save sample'):
        dataset_name = st.text_input('Dataset Name')
        description = st.text_area('Description', value=f'{sampling_strategy.capitalize()} sample of {dataset.upload_name} <ID={dataset.id}>')
        origin = st.text_area('Origin', value=dataset.data.get('origin'))
        LIC = components.utils.LICENSES
        license = st.selectbox('License', options=list(LIC.keys()), format_func=lambda k: LIC.get(k))


        save = st.form_submit_button()
    
    if save:
        dataset = api.set_upload_data(
            dataset_name,
            'sample',
            field_id=dataset.id,
            x=[c[0] for c in coords],
            y=[c[1] for c in coords],
            v=values,
            description=description,
            origin=origin,
            license=license
        )

        # create thumbnail
        dataset.update_thumbnail()

        # set the new dataset as active
        st.session_state.data_id = dataset.id
        st.session_state.action = 'view'
        st.experimental_rerun()


def list_datasets(api: API, container=st):
    # select a dataset
    all_names = api.get_upload_names()
    if len(all_names) == 0:
        container.warning('This database has no datasets. Please upload something.')
        st.stop()
    
    data_id = container.selectbox('DATASET', options=list(all_names.keys()), format_func=lambda k: all_names.get(k))
    dataset = api.get_upload_data(id=data_id)

    # preview data
    container.title(f"{dataset.data_type.upper()} dataset")

    # create a column layout
    left, right = container.columns((6, 4))
    if 'origin' in dataset.data:
        right.markdown(dataset.data['origin'])

    # create a preview plot
    components.dataset_plot(dataset, disable_download=False, container=left)

    # some basic stats
    stats = [
        {'Stat': 'Estimated experimental variograms', 'Value': len(dataset.variograms)},
        {'Stat': 'Total number of fitted models', 'Value': np.sum([[len(cv.models) for cv in v.conf_intervals] for v in dataset.variograms])},
    ]

    # check if there is a parent field
    if 'field_id' in dataset.data:
        stats.append({'Stat': 'Parent field id', 'Value': dataset.data['field_id']})

    right.markdown('## Related data')
    right.table(stats)


def edit_dataset(dataset: DataUpload, api: API, container = st) -> None:
    # check if edit should be canceled
    cancel = st.button('CANCEL')
    if cancel:
        st.session_state.action = 'view'
        st.experimental_rerun()

    # extract the data
    data = dataset.data

    # Build the main form
    with container.form('EDIT'):
        LIC = components.utils.LICENSES
        new_title= st.text_input('Title', dataset.upload_name)
        new_origin = st.text_area('Origin', value=data.get('origin', ''), help="Add the source of the dataset to help others cite it correctly.")
        new_description = st.text_area('Description', value=data.get('description', ''), help="Add a description of the dataset.")
        new_license = st.selectbox('License', options=list(LIC.keys()), index=list(LIC.keys()).index(data.get('license', 'ccby')), format_func=lambda k: LIC.get(k))
        save = st.form_submit_button('SAVE')
    
        # check save
        if save:
            updates = {'license': new_license}
            if new_origin.strip() != '':
                updates['origin'] = new_origin
            if new_description.strip() != '':
                updates['description'] = new_description
            
            # overwrite dataset 
            dataset = api.update_upload_data(id=dataset.id, name=new_title, **updates)
            
            # switch back to view
            st.session_state.action = 'view'
            st.experimental_rerun()


def main_app(api: API):
    # check if the action state is set
    if not 'action' in st.session_state:
        dataset_grid(api=api)
    
    # we have a state
    action = st.session_state.action
    if action == 'view':
        action_view(api=api)
    elif action == 'new':
        upload_view(api=api)
    elif action == 'edit':
        edit_view(api=api)
    elif action == 'delete':
        delete_view(api=api)
    elif action == 'sample':
        sample_view(api=api)


if __name__=='__main__':
    def run(db_name='data.db'):
        api = API(db_name=db_name)
        main_app(api=api)

    import fire
    fire.Fire(run)
    