from typing import List, Union
import streamlit as st
import numpy as np
from time import time

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import VarioModelResult, VarioParams, VarioConfInterval, VarioModel, DataUpload
from skgstat_uncertainty import components


def model_table(models: List[VarioModel]) -> None:
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
    # check if the sample has a field id otherwiese use the dataset id
    field_id = dataset.data.get('field_id', dataset.id)
    
    # filter for existing auxiliary data
    aux = api.filter_auxiliary_data(parent_id=field_id)

    # if no auxiliary fields are found, just return
    if len(aux) == 0:
        return None
    else:
        return aux
    # we have auxiliary data
    info = container.empty()
    info.info(f"A total of {len(aux)} auxiliary datasets were found in the database. Select if and how many should be used. This will activate Kriging with external drift.")

    # build up an info dict
    INFO = {d.id: d.upload_name  for d in aux}
    using_ids = container.multiselect(
        'Select drift datasets for EDK',
        options=list(INFO.keys()),
        format_func=lambda k: INFO.get(k)
    )

    # check how many are selected
    if len(using_ids) == 0:
        return None
    else:
        info.empty()
        return [dataset for dataset in aux if dataset.id in using_ids]


def choose_algorithm(aux: Union[List[DataUpload], None], vario: VarioParams, container=st) -> dict:
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
        info_text.write('\n'.join(prog_text))
    
    end_time = time()
    runtime = end_time - start_time
    st.success(f"Finished after {round(runtime)} seconds.")

    return all_fields


def main_app(api: API) -> None:
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
