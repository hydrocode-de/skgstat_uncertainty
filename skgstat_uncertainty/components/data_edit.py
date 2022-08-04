from typing import Union
import streamlit as st

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty import components

def edit_dataset(dataset: DataUpload, api: API, edit: bool = True, add_preview: bool = False, container = st) -> Union[None, DataUpload]:
    """
    Create a edit form for the given dataset. It returns True, as
    the editing is finished (Save button clicked) and False otherwise.
    If you the form is used
    """
    # extract the data
    data = dataset.data

    if add_preview:
        with container.expander('PREVIEW'):
            components.dataset_plot(dataset, disable_download=True)

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
            if edit:
                dataset = api.update_upload_data(id=dataset.id, name=new_title, **updates)
            else:
                dataset = api.set_upload_data(dataset.upload_name, dataset.data_type, **{**data, **updates})
                dataset.update_thumbnail()
            
            return dataset
        else:
            return None
