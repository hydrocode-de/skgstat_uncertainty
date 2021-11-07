from typing import Union, Tuple
import streamlit as st
import os
import json
import numpy as np

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload, VarioParams, VarioConfInterval

def upload_handler(api: API, can_select=True, upload_mime=['csv', 'asc', 'txt', 'json'], container=st) -> DataUpload:
    # get all existing upload names
    all_names = api.get_upload_names(data_type=['field', 'sample'])

    # check if there is already data available
    if len(all_names) > 0 and can_select:
        do_upload = container.checkbox('Upload new data anyway', value=False)

        # if no upload wanted, choose existing data
        if not do_upload:
            selected_dataset = container.selectbox(
                'Select existing dataset',
                options=['select a dataset', *list(all_names.values())]
            )
            # load the dataset from the DB
            if selected_dataset == 'select a dataset':
                st.stop()
            dataset = api.get_upload_data(name=selected_dataset)
    else:
        do_upload = True

    # if uploading is required -> upload and restart
    if do_upload:
        msg = "Upload data to the application. The following MIME types are supported:\n\n"
        if 'csv' in upload_mime:
            msg += "\r* _csv_: A csv file containing the headers 'x' and 'y' for coordinates and 'v' for values\n"
        if 'asc' or 'txt' in upload_mime:
            msg += "\r* _asc_, _txt_: a space delimeted file of a 2D field (rows x cols)\n"
        if 'json' in upload_mime:
            msg += "\r* _json_: JSON file of a raw database dump - __experimental__."
        msg += "\n\n"

        # display message
        container.markdown(msg)

        uploaded_file = container.file_uploader(
            'Choose the data', upload_mime)

        if uploaded_file is not None:
            data_name, mime = os.path.splitext(uploaded_file.name)

            if mime == '.csv':
                data = pd.read_csv(uploaded_file)

                if not 'x' in data.columns and 'y' in data.columns and 'v' in data.columns:
                    container.error(
                        'CSV files need to specify the columns x and y for coordinates and v for values.')
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
            
            elif mime == '.json':
                data = json.load(uploaded_file)

                # check type
                if 'x' in data and 'y' in data and 'v' in data:
                    type_ = 'sample'
                elif 'field' in data:
                    type_ = 'field'
                else:
                    type_ = 'generic'

                # save the data
                dataset = api.set_upload_data(
                    data_name,
                    type_,
                    **data
                )

            else:
                container.error(f'File of type {mime} not supported.')
                st.stop()
        else:
            # stop until upload de-selected or file upload completed
            st.stop()

    return dataset


# define the return type for data selector
SELECTED = Union[
    DataUpload,
    Tuple[DataUpload,VarioParams],
    Tuple[DataUpload,VarioParams,VarioConfInterval],
]
def data_selector(api: API, stop_with: str = '', data_type='sample', container=st) -> SELECTED:
    # create the expander
    expander = container.expander('DATA SELECT', expanded=True)

    # get the different data names - only for samples
    DATA_NAMES = api.get_upload_names(data_type=data_type)
    data_id = expander.selectbox('Sample dataset', options=list(DATA_NAMES.keys()), format_func=lambda k: f'{DATA_NAMES.get(k)} <{k}>')
    dataset = api.get_upload_data(id=data_id)
    
    # check if that is all we need
    if stop_with == 'data':
        return dataset

    # filter variograms by data_id
    variograms = {v.id: v for v in api.filter_vario_params(data_id=data_id)}
    if len(variograms) == 0:
        expander.warning(f"There are no variograms estimated for dataset '{DATA_NAMES.get(data_id)}'")
        st.stop()
    vario_id = expander.selectbox('Variogram', options=list(variograms.keys()), format_func=lambda k: f'{variograms.get(k).name} <{variograms.get(k).name}>')
    vario = variograms.get(vario_id)

    if stop_with == 'params':
        return dataset, vario

    # load the intervals
    intervals = {cv.id: cv for cv in vario.conf_intervals}

    if len(intervals) == 0:
        expander.warning(f"No confidence intervals are estimated for variogram '{vario.name}'")
        st.stop()

    conf_id = expander.selectbox('Confidence Interval', options=list(intervals.keys()), format_func=lambda k: f'{intervals.get(k).name} <{intervals.get(k).id}>')
    interval: VarioConfInterval = intervals[conf_id]

    return dataset, vario, interval


def upload_auxiliary_data(dataset: DataUpload, api: API) -> None:
    # enable upload for auxiliary data
    msg_area = st.empty()
    enable = st.checkbox("Upload auxiliary data", value=False)

    if not enable:
        msg_area.markdown("""Check the box below to upload auxiliary information for the dataset.""")
        return

    # show the upload
    msg_area.markdown("""The field needs to have the same resolution as other auxiliary fields and has to match the final
        resolution of the kriging field. If the sample was derived from an original field, this resolution has to match as well.""")
    uploaded_file = st.file_uploader(
        'Choose the auxiliary data', ['asc', 'txt'])

    if uploaded_file is not None:
        # get the name
        data_name, mime = os.path.splitext(uploaded_file.name)

        # check if the data has a field_id
        field_id = dataset.data.get('field_id', dataset.id)
        # get the data
        data = np.loadtxt(uploaded_file)

        # upload
        aux = api.set_upload_data(
            data_name,
            'auxiliary',
            field=data.tolist(),
            parent_id=field_id
        )

        st.success(
            'Data added to the database. You can upload more or disable the upload area')
    st.stop()
