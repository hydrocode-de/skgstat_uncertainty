from typing import List, Union
import streamlit as st
import numpy as np
from time import time

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import VarioModelResult, VarioParams, VarioModel, DataUpload
from skgstat_uncertainty import components


def model_table(models: List[VarioModel]) -> None:
    """
    Helper function to create a streamlit table of all passed
    Variogram models

    Parameters
    ----------
    models : List[VarioModel]
        List of models to be displayed in the table

    """
    # if running, create the table
    table_data = []
    for model in models:
        par = dict(id=model.id)
        par.update(model.parameters['model_params'])
        par.update(model.parameters.get('measures'))
        table_data.append(par)
    
    # create the table
    st.dataframe(table_data)
    
    return models


def check_for_auxiliary_data(dataset: DataUpload, api: API, container = st) -> Union[List[DataUpload], None]:
    """
    Helper function to check for auxiliary data present for the 
    given dataset. Returns None if no data found

    Parameters
    ----------
    dataset : DataUpload
        Dataset for which auxiliary data is required.
    api : skgstat_uncertainty.api.API
        Connected instance of the SciKit-GStat Python API to interact with
        the backend.

    """
    # check if the sample has a field id otherwiese use the dataset id
    field_id = dataset.data.get('field_id', dataset.id)
    
    # filter for existing auxiliary data
    aux = api.filter_auxiliary_data(parent_id=field_id)

    # if no auxiliary fields are found, just return
    if len(aux) == 0:
        return None
    else:
        return aux


def choose_algorithm(aux: Union[List[DataUpload], None], vario: VarioParams, container=st) -> dict:
    """
    Component to render a Kriging algorithm selection interface to the user.
    The user can select Ordinary, Simple, Universal or External drift kriging. 
    Based on the selection, different parameterization dialogs are rendered.

    Parameters
    ----------
    aux: List[DatUpload], optional
        List of auxiliary datasets used for kriging. Can be None, 
        if no auxiliary data is present, External drift kriging will be disabled.
    vario: VarioParams
        Variogram parameterization used to apply the selected kriging algorithm.
        Note, that this component only renders the dialog and does not apply the
        algorithm.
    
    Returns
    -------
    opt : dict
        Agrument dictionary to be passed to :any:`Krige <gstools.Krige>`
    
    Example
    -------
    The return of this component can directly be passed to a Kriging instance
    created by the variogram:

    >> # Get the variogram and kriging instances
    >> vario = varioParams.variogram
    >> krige = vario.to_gs_krige(**opt)

    """
    # There are different Kriging algorithms available
    available = {'ordinary': 'Ordinary Kriging', 'simple': 'Simple Kriging', 'universal': 'Universal Kriging'}
    preselect = 'ordinary'

    # if aux is given -> use EDK
    if aux is not None:
        available['edk'] = 'External drift Kriging'
        preselect = 'edk'
    
    # build the select
    option = container.selectbox('Kriging algorithm', options=list(available.keys()), format_func=lambda k: available.get(k), index=list(available.keys()).index(preselect))

    opt = dict(init_args=dict(), call_args=dict())
    # switch the options:
    # Simple kriging
    if option == 'simple':
        mean = container.number_input('Field mean', value=0.0, help="For simple kriging, the real field mean value is needed")
        opt['init_args']['mean'] = mean
    
    # external drift kriging
    elif option == 'edk':
        # create the selector for drift terms
        INFO = {d.id: d.upload_name for d in aux}
        edk_id = container.selectbox('EDK drift term', options=list(INFO.keys()), format_func=lambda k: INFO.get(k))
        
        # get the the dataset
        drift_data: DataUpload = [data for data in aux if data.id == edk_id].pop()
        
        # extract the coordinates
        coords = vario.variogram.coordinates
        drift_field = np.array(drift_data.data['field'])
        cond_drift = [drift_field[tuple(c)] for c in coords]

        # set the parameter
        opt['init_args']['ext_drift'] = cond_drift
        opt['call_args']['ext_drift'] = drift_data.data['field']
    
    # universal kriging
    elif option == 'universal':
        FUNC = {'linear': 'Linear drift', 'quadratic': 'Quadratic drift'}
        func_name = container.radio('Regional drift function', options=list(FUNC.keys()), format_func=lambda k: FUNC.get(k))
        opt['init_args']['drift_functions'] = func_name

    elif option == 'ordinary':
        opt['init_args']['unbiased'] = True
    
    return opt


def apply_kriging(models: List[VarioModel], dataset: DataUpload, vario: VarioParams, opts: dict, api: API) -> List[VarioModelResult]:
    """
    Main component to apply kriging algorithm. The component renders all needed dialogs
    and filters. The same kriging algorithm can be applied to a list of models and their
    parameterizations at once. The progress will be displayed by a number of progress bars.

    Parameters
    ----------
    models : List[VarioModel]
        List of parameterized theoretical model instances. Each will be used for a kriging
        application.
    dataset : DataUpload
        Base dataset, which will be used as the conditional grid points for the kriging.
    vario : VarioParams
        Basic empirical Variogram representation, which is shared by all parameterized
        model instances.
    opts : dict
        Kriging parameters
    api : skgstat_uncertainty.api.API
        Connected instance of the SciKit-GStat Python API to interact with
        the backend.

    Note
    ----
    This component restarts and stops the streamlit application on user interaction.

    """
    st.markdown("""## Apply models""")
    st.markdown("""Kriging can take a time. Therefore you need to start the processing manually using the button below below.""")

    # get the number of models
    n = len(models)

    # check how many results are already there
    existing = {}
    for model in models:
        try:
            res = api.filter_results(model_id=model.id).pop()
        except IndexError:
            continue
        existing[model.id] = res
    
    # check if anything already exists:
    if len(existing) > 0:
        drop_and_recalc = st.checkbox(f'Overwrite {len(existing)} existing kriging results', value=False)
        
        # use cached results if needed
        if len(existing) == n and not drop_and_recalc:
            return list(existing.values())

        if drop_and_recalc:
            # delete the results
            for res in list(existing.values()):
                api.delete_result(id=res.id)
                st.success('Fields deleted')
            existing = {}
    
    # check if still existing
    if len(existing) > 0:
        st.info(f"{len(existing)} kriging fields already found in the database")

    start = st.button('RUN KRIGING NOW')
    if not start:
        # do whatever needed here
        st.stop()
    
    # we actually run kriging now
    # define the grid bounds
    if opts['call_args'].get('ext_drift', False):
        _x, _y = np.array(opts['call_args']['ext_drift']).shape
    elif dataset.data.get('field_id', False):
        parent_dataset = api.get_upload_data(id=dataset.data.get('field_id'))
        parent_field = parent_dataset.data['field']
        _x, _y = np.array(parent_field).shape
    else:
        c = vario.variogram.coordinates
        _x = np.max(c[:,0])
        _y = np.max(c[:,1])
    
    # create the grid
    x = range(_x)
    y = range(_y)

    progress = st.progress(0)
    prog_text = []
    info_text = st.empty()
    start_time = time()
    all_fields: List[VarioModelResult] = []

    for i in range(n):
        model = models[i]

        # check if this model already exists
        if model.id in existing:
            all_fields.append(existing[model.id])
            prog_text.append(f"[{i + 1}/{n}]: cached with ID {existing[model.id].id}")
            info_text.write('\n'.join(prog_text))
            continue

        # get the variogram of that model
        variogram = model.variogram

        # build the kriging algorithm
        krige = variogram.to_gs_krige(**opts['init_args'])
        
        # run 
        t1 = time()
        result, sigma = krige([x, y], mesh_type='structured', **opts['call_args'])
        t2 = time()

        # save result
        res = api.set_result(model.id, 'kriging_field', field=result.tolist(), sigma=sigma.tolist(), options=opts, runtime=(t2 - t1))
        all_fields.append(res)

        # update the bar
        progress.progress((i + 1) / n)
        prog_text.append(f"[{i + 1}/{n}]: kriging finsihed after {round(t2 - t1)} seconds.")
        info_text.markdown('\n\n'.join(prog_text))
    
    end_time = time()
    runtime = end_time - start_time
    st.success(f"Finished after {round(runtime)} seconds.")

    return all_fields


def main_app(api: API) -> None:
    """
    Kriging chapter.
    This streamlit application can be run on its own or embedded into another
    application. The application displays a datasets selection dialog to load
    the base dataset. Before the user is able to apply kriging algorithms to
    the dataset, an empirical variogram has to be estimated and at least one
    theoretical variogram model has to be parameterized.

    Parameters
    ----------
    api : skgstat_uncertainty.api.API
        Connected instance of the SciKit-GStat Python API to interact with
        the backend.

    """
    st.title('Model application by Kriging')
    st.markdown("""This chapter aims at generating result field by interpolating the original sample 
        using the models fitted within the confidence intervals.
    """)

    # load the dataset, and interval to be used
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)

    # enable upload if needed
    components.upload_auxiliary_data(dataset=dataset, api=api)

    # sidebar container
    method_select_container = st.sidebar.expander('METHOD SELECTION', expanded=True)
    
    # select drift if aux data available
    aux = check_for_auxiliary_data(dataset=dataset, api=api, container=method_select_container)

    # build the method control options
    opts = choose_algorithm(aux=aux, vario=vario, container=method_select_container)
    
    # load all models from the interval
    models: List[VarioModel] = interval.models

    if len(models) == 0:
        st.warning(f"There are no models fitted within the {interval.name} confidence interval of {interval.variogram.name} variogram.")
        st.stop()

    # create the overview table
    models = model_table(models)

    # apply kriging
    kriging_fields = apply_kriging(models=models, dataset=dataset, vario=vario, opts=opts, api=api)
    
    # create the exander for the result 
    # field_box = st.expander('KRIGING FIELDS', expanded=False)
    # components.multi_plot_field_heatmaps(kriging_fields, container=field_box)
    
    st.success("All fields are interpolated. You can now continue with any analyzing chapter.")
    st.stop()

if __name__ == '__main__':
    api = API()
    main_app(api)
