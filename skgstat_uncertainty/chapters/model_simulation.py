from typing import List, Dict
import streamlit as st
import numpy as np
import gstools as gs
import time
import plotly.graph_objects as go

from skgstat_uncertainty.api import API
from skgstat_uncertainty import components
from skgstat_uncertainty.models import VarioModel


def run_single_simulation(model: VarioModel, N: int = 100, show_progess: bool = True, seed: int = 42, container=st) -> np.ndarray:
    """
    Run a geostatistical simulation for one model parameterization. 
    The simulation is done using GSTools conditional spatial random field class
    :any:`CondSRF <gstools.CondSRF>`. A progess bar will indicate the 
    progress to the user. The component returns a 3-dimensional numpy array, containing
    all simulations on the 3rd axis.

    Parameters
    ----------
    model : VarioModel
        Instance of a variogram model parameterization used to condition the field.
    N : int
        Number of iterations. This also determines the length of the
        returns 3rd axis.
    show_progress : Bool
        If True (default), the component will render a progress bar
    seed : int
        Any integer to seed the random number generator used for the 
        stochastic simulation.

    """
    # extract the variogram
    variogram = model.variogram

    # get the grid
    grid = model.get_base_grid()

    # initialize a Kriging instance to condition the SRF
    krige = variogram.to_gs_krige()
    cond_srf = gs.CondSRF(krige)

    # create a container for the results
    fields = []

    # create a progress bar if needed
    if show_progess:
        progress_bar = container.progress(0)

    for i in range(N):
        # run a simulation
        field = cond_srf.structured(tuple([range(dim) for dim in grid]), seed=seed+i)
        fields.append(field)

        # update the progress bar
        if show_progess:
            progress_bar.progress((i + 1) / N)
    
    # finished
    return np.stack(fields, axis=2)


def run_simulations(simulations: Dict[int, np.ndarray], models: List[VarioModel], opts_container=st) -> None:
    """
    Start geostatistical simulations for all passed model parameterizations.
    The component will render controls to specify the number of simulation runs per
    parameterization and seeding. The progress of the simulations is indicated by a progress bar.

    Parameters
    ----------
    simulations : dict
        dictionary of all model runs. If there are already model runs present
        the user can decide to overwrite these.
    models : List[VarioModel]
        List of variogram model parameterizations to use for the geostatistical simulation.
        The settings for simulation will be applied to each parameter set.

    Notes
    -----
    This component will restart and terminate the streamlit application on user interaction.

    """
    # add options to control simulation
    n = opts_container.number_input('Number of simulations', value=10, min_value=1, max_value=100)
    seed = opts_container.number_input('Random seed', value=42, min_value=0, max_value=9999999, help='Set a seed to make the simulations reproducible')
    
    # rerun option
    if len(simulations) > 0:
        rerun = opts_container.checkbox(f'Overwrite {len(simulations)} simulations', value=False)
    else:
        rerun = True
    
    # start button
    start = opts_container.button('START simulation')
    
    if start:
        st.markdown('### simulations')
        msg = st.empty()
        progress_bars = st.container()
        
        runtimes = []
        for i, model in enumerate(models):
            msg.info(f'[{i + 1}/{len(models)}] Estimated runtime: {(np.mean(runtimes) * (len(models) - (i + 1))).round(1)} seconds')
            if model.id not in simulations or rerun:
                # run the simulation
                t1 = time.time()
                stack = run_single_simulation(model=model, N=n, show_progess=True, container=progress_bars, seed=seed)
                t2 = time.time()
                runtimes.append(t2 - t1)

                # save the simulation
                st.session_state.simulations[model.id] = stack
            else:
                progress_bars.progress(1.0)

        # save the used options to the session
        st.session_state.last_opts = {'type': 'simulation', 'N': n, 'seed': seed}
        # rerun
        st.experimental_rerun()
    else:
        st.stop()


def save_results(simulations: Dict[int, np.ndarray], models: List[VarioModel], api: API) -> None:
    """
    Main component to store simulation results into the database.
    All simulations for each model parametrization are aggregated to their mean value 
    per cell and the cell variance. The data type for stored results are 'simulation_field'.

    Parameters
    ----------
    simulations : dict
        dictionary of all model runs. The dictionary used the parameter set ID as key
        and a stacked numpy array of all model runs for this parameter.
    models : List[VarioModel]
        List of variogram model parameterizations used for the geostatistical simulation.
        The list must be provided to link the new database entries for the simulation results
        with the parameter sets, they originate from.
    api : skgstat_uncertainty.api.API
        Connected instance of the SciKit-GStat Python API to interact with
        the backend.

    """
    # all simulations done, show results
    st.success('All simulations done')
    st.markdown('### Save results\n The results are not saved to the database yet!')
    st.markdown('You can use the controls to inspect the simulation results and save them as a new result set. By clicking SAVE, **all fields** will be saved.')
    
    # build the lookup for model selection
    MODS = {model.id : f"{model.model_type.capitalize()} model <ID={model.id}>" for model in models}
    model_id = st.selectbox('Select a model', options=list(MODS.keys()), format_func=lambda k: MODS.get(k))
    
    field_mean = np.mean(simulations[model_id], axis=2)
    field_std = np.std(simulations[model_id], axis=2)

    # show 
    left, right = st.columns(2)
    # Colorscales
    CS = ['Blackbody', 'Bluered', 'Blues','Cividis', 'Earth', 'Electric', 'Greens', 'Greys', 'Hot', 'Jet', 'Picnic','Portl', 'Rainbow', 'RdBu', 'Reds', 'Viridis', 'YlGnBu', 'YlOrRd']

    # create figures
    with left:
        cm1 = st.selectbox('Color map', options=CS, key='fig1_cm')
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=field_mean, colorscale=cm1))
        fig.update_layout(
            title='Simulation Mean',
            yaxis=dict(scaleanchor='x')
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cm2 = st.selectbox('Color map', options=CS, index=15, key='fig2_cm')
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=field_std, colorscale=cm2))
        fig.update_layout(
            title='Simulation Std',
            yaxis=dict(scaleanchor='x')
        )
        st.plotly_chart(fig, use_container_width=True)

    # save the results
    save1 = st.sidebar.button('SAVE RESULTS', key="save1")
    save2 = st.button('SAVE RESULTS', key="save2")

    if save1 or save2:
        with st.spinner('Saving results to database...'):
            # options
            opts = st.session_state.last_opts

            # save all results
            for model_id, stack in simulations.items():
                field = np.mean(stack, axis=2).tolist()
                std = np.std(stack, axis=2).tolist()
                api.set_result(model_id, 'simulation_field', field=field, std=std, options=opts)

            st.success('Simulation results saved')
    st.stop()


def main_app(api: API):
    """
    Geostatistical simulation chapter.
    This streamlit application can be run on its own or embedded into another
    application. This chapter is build around geostatistical simulations.
    For any variogram parameter set the user can setup a geostatistical simulation.
    The single simulated fields will be aggregated to the simulations' mean value
    and can be saved back to the database and used as a field dataset.

    Parameters
    ----------
    api : skgstat_uncertainty.api.API
        Connected instance of the SciKit-GStat Python API to interact with
        the backend.

    Notes
    -----
    This chapter requires an estimated empirical variogram instance and at least one
    variogram model parameter set to be present in the database.

    """

    st.title('Model Simulation')
    st.markdown("This chapter is an alternative to the Kriging chapter. Instead of predicting the target field, it will run geostatistical simulations.")

    # load the dataset and interval to be used
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)
    models = interval.models

    # ids used on this run
    valid_model_ids = [model.id for model in models]

    # check if there are already simulations
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    simulations = {k: v for k, v in st.session_state.get('simulations', {}).items() if k in valid_model_ids}

    # create a table anchor
    table_anchor = components.model_table(models=models, variant='dataframe')

    # check if simulations are missing
    if len(simulations) < len(models):
        st.info(f" Found {len(simulations)} simulations. First run the missing {len(models) - len(simulations)} simulations to continue.")
        # container for simulation options
        sim_exp = st.sidebar.expander('SIMULATION SETTINGS', expanded=True)
        
        # run the simulations
        run_simulations(simulations=simulations, models=models, opts_container=sim_exp)

    elif len(simulations) == len(models):
        save_results(simulations=simulations, models=models, api=api)


if __name__ == "__main__":
    st.set_page_config(page_title='Model Simulation', layout='wide')
    
    def run(data_path=None, db_name='data.db'):
        api = API(data_path=data_path, db_name=db_name)
        main_app(api)
    
    import fire
    fire.Fire(run)
