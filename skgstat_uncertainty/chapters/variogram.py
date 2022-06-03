from typing import Tuple
from unittest.main import MAIN_EXAMPLES
import streamlit as st
import numpy as np

import plotly.graph_objects as go
from skgstat import Variogram

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty import components
from skgstat_uncertainty.components.utils import BIN_FUNC, ESTIMATORS, MAXLAG, CONF_METHODS
from skgstat_uncertainty.processor import exp_variogram_uncertainty as variogram_processor


CONF_INTRO = """### Calculate confidence interval

You can now estimate a confidence interval for your empirical variogram. Keep in mind that
an empirical variogram is sample statistics and therefore you need a confidence interval
to use parameterize a theoretical variogram function and apply the model.

Do you want to activate the confidence interval estimation or just save the empirical variogram
without confidence interval?
"""

def variogram_manual_fit(dataset: DataUpload, container=st.sidebar) -> Variogram:
    """
    """
    # create the options and parameters
    # ---------------

    # binning
    bin_func = container.selectbox('Binning method', options=list(BIN_FUNC.keys()), format_func=lambda k: BIN_FUNC.get(k))
    if bin_func in ('even', 'uniform', 'kmeans', 'ward'):
        n_lags = int(container.number_input('Number of lag classes', value=10, min_value=3, step=1))
    else:
        # default
        n_lags = 10
    
    # estimator
    estimator = container.selectbox('Semi-variance estimator', options=list(ESTIMATORS.keys()), format_func=lambda k: ESTIMATORS.get(k))
        
    # maxlag settings
    maxlag_type = container.selectbox('Maxlag settings', options=list(MAXLAG.keys()), format_func=lambda k: MAXLAG.get(k))
    if maxlag_type == 'none':
        maxlag = None
    elif maxlag_type == 'absolute':
        maxlag = container.number_input('Absolute maxlag', value=50, min_value=0)
    elif maxlag_type == 'ratio':
        maxlag = container.slider('Ratio of max value', value=0.6, min_value=0., max_value=1.)
    else:
        maxlag = maxlag_type

    # get the coordinates and values
    coords = list(zip(*[dataset.data.get(dim) for dim in ('x', 'y', 'z') if dim in dataset.data]))
    values = dataset.data['v']

    # all set -> estimate the variogram
    try:
        vario = Variogram(coords, values, estimator=estimator, bin_func=bin_func, n_lags=n_lags, maxlag=maxlag)
    except Exception as e:
        st.info('Sorry. The chosen parameters did not result in a valid variogram. See below for the actual error')
        err_expander = st.expander('ERROR DETAILS')
        err_expander.exception(e)
        st.stop()

    return vario


def calculate_confidence_interval(variogram: Variogram, expander_container: st.sidebar) -> Tuple[Tuple[Tuple[float, float]], dict]:
    """
    """
    # switch the propagation method
    conf_method = expander_container.selectbox('Propagation method', options=list(CONF_METHODS.keys()), format_func=lambda k: CONF_METHODS.get(k))
    st.write(conf_method)
    # add parameters common to all methods
    q = int(expander_container.slider('Confidence interval [%]', value=95, min_value=50, max_value=100))
    quartiles = [100 - q, q]

    # create a container
    conf_method_params = {'quartiles': quartiles, 'conf_method': conf_method, 'method_name': CONF_METHODS.get(conf_method)}

    # switch the method
    if conf_method == 'kfold':
        interval, params = kfold_params_interface(variogram, quartiles, container=expander_container)
    elif conf_method == 'absolute':
        interval, params = monte_carlo_params_interface(variogram, quartiles, container=expander_container)
    elif conf_method == 'std':
        interval, params = std_params_interface(variogram, quartiles)
    elif conf_method == 'residual':
        interval = variogram_processor.residual_uncertainty(variogram, quartiles)
        params = {}
    
    # merge the general params with method params
    conf_method_params.update(params)
    # finally return the interval and interval params
    return interval, conf_method_params


def kfold_params_interface(variogram: Variogram, quartiles: Tuple[int, int], container=st.sidebar) -> Tuple[Tuple[Tuple[float, float]], dict]: 
    """
    Calculate a confidence interval using a k-fold statistical robustness test.
    This will break down the residuals in each lag class into k random chunks
    and calculate the semi-variance n-times with k - 1 chunks. The specified
    percentiles of the n * k repetitions are then used as confidence interval.

    Parameters
    ----------
    variogram : Variogram
        The variogram to calculate the confidence interval for.
    quartiles : Tuple[int, int]
        The two percentiles bounds which should be used.
    container : st.sidebar
        The container to place the controls to. Any valid container-like
        streamlit object is accepted. Defaults to the sidebar.

    Note
    ----
    This procedure is quite sensitive for small sample sizes and small repetitions.
    It should be used when the source of observation uncertainty is epistemic.

    """
    # place the controls
    k = container.radio('Folds', options=[3, 5, 7, 10], index=1)
    rep = container.number_input('Repitions', min_value=1, max_value=50000, value=100)
    seed = container.number_input('Seed', min_value=0, value=42, help="Seed the cross-validation for reproducible results")

    # calculate interval
    interval = variogram_processor.kfold_residual_bootstrap(variogram, k=int(k), repititions=int(rep), q=quartiles, seed=int(seed))

    # return 
    return interval, {'k': k, 'repetitions': rep, 'seed': seed}


def std_params_interface(variogram: Variogram, quartiles: Tuple[int, int]) -> Tuple[Tuple[Tuple[float, float]], dict]:
    """
    Calculate a confidence interval using the standard deviation of the residuals.
    This method is implicitly assuming normal distribution for the residuals and
    uses the Z-score of a normal distribution fitting the residuals to calculate
    a confidence interval for this distribution.

    Parameters
    ----------
    variogram : Variogram
        The variogram to calculate the confidence interval for.
    quartiles : Tuple[int, int]
        The two percentiles bounds which should be used.

    Note
    ----
    This procedure is fast and can be used for smaller sample sizes.
    But it makes strong assumptions about the residuals. They need to be normal
    distributed and their statistical dispersion actually has to correlate with
    variogram parameter uncertainty.
    It should be used when the source of observation uncertainty is epistemic.

    """
    # get the confidence level
    conf_level = quartiles[1] / 100.
    interval = variogram_processor.conf_interval_from_sample_std(variogram=variogram, conf_level=conf_level)

    # return 
    return interval, {'conf_level': conf_level}


def monte_carlo_params_interface(variogram: Variogram, quartiles: Tuple[int, int], container=st.sidebar) -> Tuple[Tuple[Tuple[float, float]], dict]:
    """
    Calculate confidence interval using a Monte-Carlo based uncertainty propagation.
    This will calculate the semi-variance of the residuals n-times, using the full 
    observation array, but substituted with a random sample of the same size.
    The statistical properties of the newly created sample can be controled by the chosen method.

    Parameters
    ----------
    variogram : Variogram
        The variogram to calculate the confidence interval for.
    quartiles : Tuple[int, int]
        The two percentiles bounds which should be used.
    container : st.sidebar
        The container to place the controls to. Any valid container-like
        streamlit object is accepted. Defaults to the sidebar.

    Note
    ----
    This method is an actual uncertainty propagation and not a approximation of the
    the confidence interval. Thus, this is the preferred method. Note that it only
    handles aleatory uncertainty and, thus can only be applied if the observation
    uncertainty can be quantified.

    """
    # define the methods
    SIG = {'precision': 'error bounds', 'sem': 'standard error of mean', 'std': 'standard deviation'}
    sigma_type = container.selectbox('Uncertainty measure type', options=list(SIG.keys()), format_func=lambda k: SIG.get(k), help="Specify the meaning of the")
    
    # switch the type to get a sigma
    if sigma_type == 'sem':
        sigma = container.number_input(f'Observation standard error of mean', min_value=0.0, value=0.001 * np.mean(variogram.values), step=0.0005 * np.mean(variogram.values))
    elif sigma_type == 'std':
        sigma = container.number_input("Observation's standard deviation", min_value=0.0, value=0.05*np.std(variogram.values), step=0.01*np.std(variogram.values))
    else:
        sigma = container.number_input("Absolute observation precision", min_value=0.0, value=np.min([d for d  in np.diff(variogram.values) if d > 0]))
            
    # add the params unique to all three methods
    rep = container.number_input('Simulations', min_value=10, max_value=100000, value=500)
    seed = container.number_input('Seed', min_value=0, value=42, help="Seed the cross-validation for reproducible results")

    # check if MC simulation is started
    if not st.session_state.get('start_mc', False):
        st.warning('MonteCarlo simulation can take some time, therefore you have to start manually. Use the button below.')
        start = st.button('START')
        if start:
            st.session_state['start_mc'] = True
            st.experimental_rerun()
        else:
            st.stop()

    # HERE start the simulation

    # create the two columns for for visualization
    mc_left, mc_right = st.columns(2)
    convergence_plot = mc_left.empty()
    confidence_plot = mc_right.empty()

    # run monte carlo simulation
    last_result = []
    divider = int(rep / 100) if rep / 100 > 1 else 1
    for i, intermediate_result in enumerate(variogram_processor.mc_absolute_observation_uncertainty(variogram, sigma=sigma, iterations=rep, seed=seed, sigma_type=sigma_type)):
        # get the last result
        last_result = intermediate_result
        
        # this is rendered really often:
        if i % divider == 0 and i > 0:  # render only every 100 steps
            # convergence plot
            conv = [np.median(intermediate_result[:j, :], axis=0) - variogram.experimental for j in range(i)]
            x  = list(range(i))
            conv_fig = go.Figure()
            for col in range(len(variogram.bins)):
                conv_fig.add_trace(go.Scatter(x=x, y=[conv[_x][col] for _x in x], mode='lines', line=dict(color='green')))
            conv_fig.update_layout(showlegend=False)
            convergence_plot.plotly_chart(conv_fig)

            # confidence interval plot
            conf_low = np.min(intermediate_result[:i, :], axis=0)
            conf_high = np.max(intermediate_result[:i, :], axis=0)
            conf_fig = go.Figure()
            conf_fig.add_trace(go.Scatter(x=variogram.bins, y=conf_low, mode='lines', fill=None, line_color='grey'))
            conf_fig.add_trace(go.Scatter(x=variogram.bins, y=conf_high, mode='lines', fill='tonexty', line_color='grey'))
            confidence_plot.plotly_chart(conf_fig)
                
    # after loop finished calculate interval
    interval = list(zip(
        np.percentile(last_result, q=quartiles[0], axis=0),
        np.percentile(last_result, q=quartiles[1], axis=0)
    ))

    # return
    return interval, {'sigma': sigma, 'sigma_type': sigma_type, 'repetitions': rep, 'seed': seed}


def load_or_estimate_variogram(dataset: DataUpload, api: API, expander_container=st.sidebar) -> Variogram:
    """
    Either load existing variogram or estimate a new one
    """
    # the user has to make a decision - load available params if any
    available_vario_names = {vparam.id: vparam.name for vparam in api.filter_vario_params(data_id=dataset.id)}
        
    # check the amount of available variograms
    if len(available_vario_names) > 0:
            # check if there is a vario_id in the session
        if 'vario_id' in st.session_state:
            if st.session_state.vario_id not in available_vario_names:
                del st.session_state.vario_id
            expander_container.selectbox('Select variogram', options=list(available_vario_names.keys()), format_func=lambda k: f"{available_vario_names.get(k)} <ID={k}>", key='vario_id')
            vparam = api.get_vario_params(id=st.session_state.vario_id)
            return vparam.variogram
        else:
            omit_estimation = st.checkbox('Use existing variogram parameters', value=False)
    else:
        omit_estimation = False
    
    # check if an estimation is needed
    if not omit_estimation:
        emp_expander = st.sidebar.expander('VARIOGRAM HYPER-PARAMETERS', expanded=True)
        variogram = variogram_manual_fit(dataset=dataset, container=emp_expander)
    else:
        # otherwise select a model
        left, right = st.columns((7,2))
        vario_id = left.selectbox('Select existing empirical variogram', options=list(available_vario_names.keys()), format_func=lambda k: f"{available_vario_names.get(k)} <ID={k}>")
    
        # add a preview
        vparam = api.get_vario_params(id=vario_id)
        variogram = vparam.variogram

        fig = plot_variogram_params(variogram)
        st.plotly_chart(fig, use_container_width=True)

        # add the load button
        right.markdown("""<br>""", unsafe_allow_html=True)
        load = right.button('LOAD')
        if load:
            st.session_state.vario_id = vario_id
            st.experimental_rerun()
        else:
            st.stop()

    # finally return
    return variogram


def plot_variogram_params(variogram: Variogram, interval: Tuple[Tuple[float, float]] = None, interval_params: dict = {}, container=st.sidebar) -> go.Figure:
    """Create a preview
    """
    # build the basic plot
    fig = go.Figure(go.Scatter(
        x=variogram.bins,
        y=variogram.experimental,
        mode="markers",
        marker=dict(size=10, color='#8c24c7'),
        name="Experimental variogram"
    ))

    # handle intervals
    if interval is not None:
        # get the quartiles
        quartiles = interval_params.get('quartiles', [5, 95])
        
        # add the confidence interval
        fig.add_trace(go.Scatter(
            x=variogram.bins,
            y=[b[0] for b in interval],
            mode='lines',
            line_color='#9942cb',
            fill=None,
            name=f"{quartiles[0]}% percentile"
        ))

        fig.add_trace(go.Scatter(
            x=variogram.bins,
            y=[b[1] for b in interval],
            mode='lines',
            line_color='#9942cb',
            fill='tonexty',
            name=f"{quartiles[1]}% percentile"
        ))
    
        use_log = container.checkbox('Log-scale', value=False)
        if use_log:
            fig.update_layout(yaxis=dict(type="log"))
        else:
            fig.update_yaxes(range=[0, 1.05 * np.nanmax(np.array(interval))], title=f"{variogram.estimator.__name__} semi-variance")
    else:
        fig.update_yaxes(range=[0, 1.1 * np.nanmax(variogram.experimental)], title=f"{variogram.estimator.__name__} semi-variance")
    
    # build figure
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h')
    )

    return fig


def save_handler(dataset: DataUpload, variogram: Variogram, interval: Tuple[Tuple[float, float]] = None, interval_params: dict = {}) -> None:
    """Activate controls to save the variogram"""
    # first check if the variogram is new
    vario_is_new = 'vario_id' not in st.session_state
    
    # create the correct text
    if vario_is_new:
        st.markdown("You can save this variogram to the database to use it in the next chapters. Specify a name and hit *save*.")
        if interval is None:
            st.warning("This Variogram has no associated uncertainties estimates. Without these, you can't use it in the next chapters.")
    
    # has interval ?
    if interval is not None:
        st.markdown("The new confidence intervals need a name to store them to the database")

    # build a form
    with st.form('save_form'):
        if vario_is_new:
            st.markdown('#### Variogram')
            vario_name = st.text_input('Alias for this Variogram')
            vario_description = st.text_area('Description')
            if vario_description.strip() == "":
                vario_description = None
        
        if interval is not None:
            st.markdown('#### Confidence interval')
            conf_name = st.text_input('Alias for the confidence interval of variogram')  

        do_save = st.form_submit_button('SAVE')

    # stop is save was not hit
    if not do_save:
        st.stop()

    # Save the variogram
    if vario_is_new:
        vario = api.set_vario_params(
            name=vario_name,
            params=variogram.describe()['params'],
            data_id=dataset.id,
            description=vario_description
        )
        st.success(f'Saved variogram {vario.name} with ID {vario.id}.')
    else:
            # pass the formerly loaded vparam obj
            vario = api.get_vario_params(id=st.session_state.vario_id)

    # Save the confidence interval
    if interval is not None:
        conf = api.set_conf_interval(
            name=conf_name,
            vario_id=vario.id,
            interval=interval,
            **interval_params
        )
        st.success(f'Saved confidence interval {conf.name} with ID {conf.id}')
    st.button('RELOAD')


def main_app(api: API) -> None:
    st.title('Variogram estimation')
    
    main_params_exp = st.sidebar.expander('OPTIONS', expanded=True)
    
    # handle data select    
    if 'data_id' not in st.session_state:
        components.data_select_page(api)
    else:
        dataset = components.data_selector(api, stop_with='data', data_type='sample', container=main_params_exp, add_expander=False)

    # place a title
    st.markdown(f'## Estimate a variogram for the {dataset.upload_name} dataset')
    # load the variogram
    variogram = load_or_estimate_variogram(dataset, api, expander_container=main_params_exp)

    # check if the user wants to calculate a confidence interval
    if not 'calculate_interval' in st.session_state:
        st.markdown(CONF_INTRO)
        
        # add the buttons
        left, right = st.columns(2)
        use = left.button('ENABLE CONFIDENCE INTERVAL')
        omit = right.button('DISABLE CONFIDENCE INTERVAL')

        if use:
            st.session_state.calculate_interval = True
            st.experimental_rerun()
        elif omit:
            st.session_state.calculate_interval = False
            st.experimental_rerun()
        else:
            interval = None
            interval_params = {}
            # st.stop()
    else:
        main_params_exp.checkbox('Enable confidence calculation', key='calculate_interval')
    
    if st.session_state.calculate_interval:
        conf_box = st.sidebar.expander('CONFIDENCE INTERVAL SETTINGS', expanded=True)
        interval, interval_params = calculate_confidence_interval(variogram, expander_container=conf_box)
    else:
        interval = None
        interval_params = {}

    # plot the variogram
    preview_box = st.expander('VARIOGRAM PREVIEW', expanded=True)
    opt, plot_area = preview_box.columns((2, 8))
    opt.markdown('#### Options')
    fig = plot_variogram_params(variogram, interval, interval_params, container=opt)
    plot_area.plotly_chart(fig, use_container_width=True)

    # handle save
    with st.expander('SAVE VARIOGRAM'):
        save_handler(dataset, variogram, interval, interval_params)


if __name__ == '__main__':
    st.set_page_config(page_title='Variogram estimation', layout='wide')
    
    def run(data_path=None, db_name='data.db'):
        api = API(data_path=data_path, db_name=db_name)
        main_app(api)
    
    import fire
    fire.Fire(run)
