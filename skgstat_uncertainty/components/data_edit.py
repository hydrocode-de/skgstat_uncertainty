from typing import Union
import streamlit as st


from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty import components


def edit_dataset(dataset: DataUpload, api: API, edit: bool = True, add_preview: bool = False, container = st) -> Union[None, DataUpload]:
    """
    Create a edit form for the given dataset. It returns the dataset, as
    the editing is finished (Save button clicked) and None otherwise.
    This form can also be used to create new datasets.

    Parameters
    ----------
    dataset : DataUpload
        The dataset instance to be added / edited.
    api : API
        The API instance to be used for persisting changes.
    edit : bool
        If True, the form is in edit mode and changes will be reflected
        as an UPDATE statement in the database. If False, the form will
        create a new entry in the database.
    add_preview : bool
        If True, an expander element previewing the dataset will be added.
        This is helpful if the form is not in edit mode, to preview data,
        before it is added.
    container : streamlit.component.Component
        Any streamlit element to form should be child of.

    Returns
    -------
    Union[None, DataUpload]
        Returns the DataUpload object if the user saved the changes
        and None otherwise

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
        
        # optional DOI
        if 'doi' in data or not edit:
            new_doi = st.text_input('DOI', value='10.', help="Add the DOI, if the dataset has an external doi.")
        else:
            new_doi = '10.'
        
        # optional CRS
        if 'CRS' in data or not edit:
            if not edit:
                use_crs = st.sidebar.checkbox('Dataset has a valid corrdinate reference system', value=False)
            else:
                use_crs = True
            
            if use_crs:
                new_crs = st.number_input('EPSG', value=4326, min_value=0, max_value=99999)
            else:
                new_crs = None
        else:
            new_crs = None
    
        # save button
        save = st.form_submit_button('SAVE')
    
        # check save
        if save:
            updates = {'license': new_license}
            if new_origin.strip() != '':
                updates['origin'] = new_origin
            if new_description.strip() != '':
                updates['description'] = new_description
            if new_doi != '10.':
                updates['doi'] = new_doi
            if new_crs is not None:
                updates['crs'] = new_crs

            # overwrite dataset
            if edit:
                dataset = api.update_upload_data(id=dataset.id, name=new_title, **updates)
            else:
                dataset = api.set_upload_data(dataset.upload_name, dataset.data_type, **{**data, **updates})
                dataset.update_thumbnail()
            
            return dataset
        else:
            return None
