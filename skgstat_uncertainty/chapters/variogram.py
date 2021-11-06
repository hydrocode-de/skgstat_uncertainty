import streamlit as st
import pandas as pd
import numpy as np
import os

import plotly.graph_objects as go
from skgstat import Variogram

from skgstat_uncertainty.api import API
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty.processor import sampling, variogram as variogram_processor
from skgstat_uncertainty import components


# create some mappings
BIN_FUNC = dict(
    even='Evenly spaced bins',
    uniform='Uniformly distributed bin sizes',
    kmeans='K-Means clustered bins',
    ward='hierachical clustered bins',
    sturges="Sturge's rule binning",
    scott="Scott's rule binning",
    sqrt="Squareroot rule binning",
    fd="Freedman-Diaconis estimator binning",
    doane="Doane's rule binning"
)

ESTIMATORS = dict(
    matheron="Matheron estimator",
    cressie="Cressie-Hawkins estimator",
    dowd="Dowd estimator",
    genton="Genton estimator",
    entropy="Shannon entropy"
)

MAXLAG = dict(
    median="Median value",
    mean="Mean value",
    ratio="Ratio of max distance",
    absolute="Absolute value",
    none="Disable maxlag"
)

CONF_METHODS = dict(
    std="Sample standard deviation inference",
    kfold="Bootstraped k-fold cross-validation",
    absolute="Observation uncertainty propagation (MC)",
    residual="Residual extrema elimination",
)


def estimate_variogram(dataset: DataUpload, api: API) -> None:
    st.markdown(f'## Estimate a variogram for the {dataset.upload_name} dataset')
    
    # load the variograms for the current dataset
    available_vario_names = {vparam.id: vparam.name for vparam in api.filter_vario_params(data_id=dataset.id)}
    
    # check if a new variogram should be estimated
    if len(available_vario_names.keys()) > 0:
        omit_estimation = st.checkbox('Disable estimation. Load existing variogram', value=False)
    else:
        omit_estimation = False

    if not omit_estimation:
        # build the sidebar
        st.sidebar.title('Variogram parameter')
        bin_func = st.sidebar.selectbox('Binning method', options=list(BIN_FUNC.keys()), format_func=lambda k: BIN_FUNC.get(k))
        if bin_func in ('even', 'uniform', 'kmeans', 'ward'):
            n_lags = st.sidebar.number_input('Number of lag classes', value=10, min_value=3, step=1)
        else:
            # default
            n_lags = 10
        estimator = st.sidebar.selectbox('Semi-variance estimator', options=list(ESTIMATORS.keys()), format_func=lambda k: ESTIMATORS.get(k))
        
        # maxlag settings
        maxlag_type = st.sidebar.selectbox('Maxlag settings', options=list(MAXLAG.keys()), format_func=lambda k: MAXLAG.get(k))
        if maxlag_type == 'none':
            maxlag = None
        elif maxlag_type == 'absolute':
            maxlag = st.sidebar.number_input('Absolute maxlag', value=50, min_value=0)
        elif maxlag_type == 'ratio':
            maxlag = st.sidebar.slider('Ratio of max value', value=0.6, min_value=0., max_value=1.)
        else:
            maxlag = maxlag_type

        # get the coordinates and values
        coords = list(zip(*[dataset.data.get(dim) for dim in ('x', 'y', 'z') if dim in dataset.data]))
        values = dataset.data['v']

        # all set -> estimate the variogram
        try:
            V = Variogram(coords, values, estimator=estimator, bin_func=bin_func, n_lags=n_lags, maxlag=maxlag)
        except Exception as e:
            st.info('Sorry. The chosen parameters did not result in a valid variogram. See below for the actual error')
            err_expander = st.expander('ERROR DETAILS')
            err_expander.exception(e)
            st.stop()
    else:
        # create a variogram loading selector
        vario_id = st.selectbox('Load Variogram', options=list(available_vario_names.keys()), format_func=lambda k: available_vario_names.get(k))
        
        vparam = api.get_vario_params(id=vario_id)
        V = vparam.variogram

    # calculate the conf interval
    calculate_interval = st.checkbox('Estimate uncertainties for experimental variogram', value=False)

    if calculate_interval:
        if omit_estimation:
            st.sidebar.title('Uncertainty Parameters')
            conf_box = st.sidebar.container()
        else:
            conf_box = st.expander('UNCERTAINTY ESTIMATION', expanded=True)
        conf_method = conf_box.selectbox('Propagation method', options=list(CONF_METHODS.keys()), format_func=lambda k: CONF_METHODS.get(k))

        q = conf_box.slider('confidence interval [%]', value=95, min_value=50, max_value=100)
        quartiles = [100 - q, q]
        
        # check wich controls are needed
        if conf_method == 'kfold':
            k = conf_box.radio('Folds', options=[3, 5, 7, 10], index=1)
            rep = conf_box.number_input('Repitions', min_value=1, max_value=50000, value=100)
            seed = conf_box.number_input('Seed', min_value=0, value=42, help="Seed the cross-validation for reproducible results")

            # calculate interval
            interval = variogram_processor.kfold_residual_bootstrap(V, k=int(k), repititions=int(rep), q=quartiles, seed=int(seed))
        
        elif conf_method == 'absolute':
            SIG = {'precision': 'error bounds', 'sem': 'standard error of mean', 'std': 'standard deviation'}
            sigma_type = conf_box.selectbox('Uncertainty measure type', options=list(SIG.keys()), format_func=lambda k: SIG.get(k), help="Specify the meaning of the")
            if sigma_type == 'sem':
                sigma = conf_box.number_input(f'Observation standard error of mean', min_value=0.0, value=0.001 * np.mean(V.values), step=0.0005 * np.mean(V.values))
            elif sigma_type == 'std':
                sigma = conf_box.number_input("Observation's standard deviation", min_value=0.0, value=0.05*np.std(V.values), step=0.01*np.std(V.values))
            else:
                sigma = conf_box.number_input("Absolute observation precision", min_value=0.0, value=np.min([d for d  in np.diff(V.values) if d > 0]))
            rep = conf_box.number_input('Simulations', min_value=10, max_value=100000, value=500)
            seed = conf_box.number_input('Seed', min_value=0, value=42, help="Seed the cross-validation for reproducible results")

            conf_box.write('MonteCarlo simulation can take some time, therefore you have to start manually.')
            start_mc = conf_box.button('Run!')

            if start_mc:
                # create the two columns for for visualization
                mc_left, mc_right = st.columns(2)
                convergence_plot = mc_left.empty()
                confidence_plot = mc_right.empty()

                # run monte carlo simulation
                last_result = []
                divider = int(rep / 100) if rep / 100 > 1 else 1
                for i, intermediate_result in enumerate(variogram_processor.mc_absolute_observation_uncertainty(V, sigma=sigma, iterations=rep, seed=seed, sigma_type=sigma_type)):
                    last_result = intermediate_result
                    # this is rendered really often:
                    if i % divider == 0 and i > 0:  # render only every 100 steps
                        # convergence plot
                        conv = [np.median(intermediate_result[:j, :], axis=0) - V.experimental for j in range(i)]
                        x  = list(range(i))
                        conv_fig = go.Figure()
                        for col in range(len(V.bins)):
                            conv_fig.add_trace(go.Scatter(x=x, y=[conv[_x][col] for _x in x], mode='lines', line=dict(color='green')))
                        conv_fig.update_layout(showlegend=False)
                        convergence_plot.plotly_chart(conv_fig)

                        # confidence interval plot
                        conf_low = np.min(intermediate_result[:i, :], axis=0)
                        conf_high = np.max(intermediate_result[:i, :], axis=0)
                        conf_fig = go.Figure()
                        conf_fig.add_trace(go.Scatter(x=V.bins, y=conf_low, mode='lines', fill=None, line_color='grey'))
                        conf_fig.add_trace(go.Scatter(x=V.bins, y=conf_high, mode='lines', fill='tonexty', line_color='grey'))
                        confidence_plot.plotly_chart(conf_fig)
                
                # after loop finished calculate interval
                interval = list(zip(
                    np.percentile(last_result, q=quartiles[0], axis=0),
                    np.percentile(last_result, q=quartiles[1], axis=0)
                ))

            else:
                st.warning('MonteCarlo simulation not yet started.')
                st.stop()
        
        elif conf_method == 'std':
            # get the confidence level
            conf_level = quartiles[1] / 100.
            interval = variogram_processor.conf_interval_from_sample_std(V, conf_level=conf_level)

        else:
            interval = variogram_processor.residual_uncertainty(V, quartiles)

    else:
        # no interval calculated
        interval = None
    
    # build the plot
    fig = go.Figure(go.Scatter(
        x=V.bins,
        y=V.experimental,
        mode="markers",
        marker=dict(size=10, color='#8effff'),
        name="Experimental variogram"
    ))
    # handle intervals
    if interval is not None:
        fig.add_trace(go.Scatter(
            x=V.bins,
            y=[b[0] for b in interval],
            mode='lines',
            line_color='#bafcfc',
            fill=None,
            name=f"{quartiles[0]}% percentile"
        ))
        fig.add_trace(go.Scatter(
            x=V.bins,
            y=[b[1] for b in interval],
            mode='lines',
            line_color='#bafcfc',
            fill='tonexty',
            name=f"{quartiles[1]}% percentile"
        ))
    
    use_log = st.sidebar.checkbox('Log-scale', value=False)
    if use_log:
        fig.update_layout(yaxis=dict(type="log"))
    elif interval is not None:
        fig.update_yaxes(range=[0, 1.05 * np.nanmax(np.array(interval))], title=f"{V.estimator.__name__} semi-variance")
    else:
        fig.update_yaxes(range=[0, 1.1 * np.nanmax(V.experimental)], title=f"{V.estimator.__name__} semi-variance")
    
    fig.update_layout(legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)

    
    if omit_estimation and interval is None:
        # variogram was loaded and no interval estimated, thus there is nothing to save
        st.stop()

    # else, there is something to save
    st.markdown('## Save')

    # first check if the variogram is new
    if not omit_estimation:
        st.markdown("You can save this variogram to the database to use it in the next chapters. Specify a name and hit *save*.")
        if interval is None:
            st.warning("This Variogram has no associated uncertainties estimates. Without these, you can't use it in the next chapters.")
    if calculate_interval:
        st.markdown("The new confidence intervals need a name to store them to the database")

    with st.form('save_form'):
        if not omit_estimation:
            st.markdown('#### Variogram')
            vario_name = st.text_input('Alias for this Variogram')
            vario_description = st.text_area('Description')
            if vario_description.strip() == "":
                vario_description = None
        if calculate_interval:
            st.markdown('#### Confidence interval')
            conf_name = st.text_input('Alias for the confidence interval of variogram')  

        do_save = st.form_submit_button('SAVE')

        if do_save:
            # reach out to api
            if not omit_estimation:
                vario = api.set_vario_params(
                    name=vario_name,
                    params=V.describe()['params'],
                    data_id=dataset.id,
                    description=vario_description
                )
                st.success(f'Saved variogram {vario.name} with ID {vario.id}.')
            else:
                # pass the formerly loaded vparam obj
                vario = vparam
            
            if calculate_interval:
                conf = api.set_conf_interval(
                    name=conf_name,
                    vario_id=vario.id,
                    interval=interval,
                    method=conf_method,
                    quartiles=quartiles
                )
                st.success(f'Saved confidence interval {conf.name} with ID {conf.id}')

        else:
            st.stop()

def main_app(api: API) -> None:
    st.title('Variogram estimation')
    
    # get or upload a dataset
    #dataset = upload_handler(api=api)
    # dataset  = components.upload_handler(api=api, container=st)

    # handle auxiliary data if needed
    # components.upload_auxiliary_data(dataset=dataset, api=api)

    # if it is a field, a subset has to be sampled
    # if dataset.data_type == 'field':
    #    sample_dense_data(dataset=dataset, api=api)
    
    # select
    dataset = components.data_selector(api=api, stop_with='data', data_type='sample')

    estimate_variogram(dataset=dataset, api=api)


    # DEV
    st.success(f"Finished. Go to the next chapter now.")


if __name__ == '__main__':
    api = API()
    main_app(api=api)
