import streamlit as st
import numpy as np
import plotly.graph_objects as go

from skgstat_uncertainty.core import Project
from skgstat_uncertainty import components
#from ..core import Project
#from .. import components

def st_app(project: Project = None) -> Project:
    # Start the application
    st.title('Fit theoretical models')
    st.markdown("""
    Use this application to fit theoretical variogram models
    that are within the error margins of the experimental
    variogram. 
    You can load experimental variograms at different levels
    of observation uncertainty and save your model parameterizations
    for later usage.
    """)
    st.sidebar.title('Parameters')

    # if the project is None, load the default one
    if project is None:
        project = Project()

    # get the correct variogram
    base_config_expander = st.sidebar.beta_expander('BASE DATA', expanded=True)
    project = components.variogram_selector(project, base_config_expander)
    project = components.simulation_family_selector(project, base_config_expander)

    # build the sigma-level dropdown
    sigma_levels = project.get_error_levels(as_dict=True)

    sigma = st.sidebar.selectbox(
        'Observation uncertainty',
        options=project.get_error_levels(),
        format_func=lambda k: sigma_levels[k],
        index=2
    )
    # sigma can possibly be None
    if sigma is None:
        st.markdown(f"""
        ### No simulations found

        The Project `{project.name}` at  `{project.path}` does not hold any Monte-Carlo simulations
        for the base variogram <{project._vario}>.
        To fit a theoretical variogram model, you need to simulate uncertainty bounds
        first.
        """)
        var_name = [v for v in project.config().get('variograms', []) if v['md5'] == project._vario][0]['name']
        st.info(f"Open OPTIONS and switch to Chapter 1 in the Navigation. Choose '{var_name}' from the stored variograms dropdown")
        st.stop()

    # update
    project.sigma = sigma

    # load the error margins
    error_bounds = project.load_error_bounds(sigma=sigma)

    # laod the variogram
    vario = project.vario

    # add the controls for the variogram
    model_name = st.sidebar.selectbox(
        'Chose a theoretical variogram function',
        options=['spherical', 'exponential', 'gaussian', 'matern', 'cubic', 'stable'],
        format_func=lambda s: s.capitalize()
    )

    # range
    r = st.sidebar.slider(
        'Specify the effective range of the model',
        min_value=0,
        max_value=int(vario.maxlag),
        value=int(np.median(vario.distance))
    )

    # nugget and sill
    nugget, sill = st.sidebar.slider(
        'Specify nugget and sill of the model',
        min_value=0,
        max_value=int(1.2 * np.max(error_bounds)),
        value=(0, int(np.mean(error_bounds)))
    )

    # shape if given
    if model_name in ('matern', 'stable'):
        shape = st.sidebar.slider(
            f'{model_name.capitalize()} shape parameter',
            min_value=0.2,
            max_value=10.,
            step=0.1,
            value=2.0
        )
    else:
        shape = None

    # do cross-validation
    always_cv = st.sidebar.checkbox(
        'Enable intermediate Cross-validation',
        value=False,
        help='If activated, a cross-validation is run on every refresh. That can take several seconds and make the application laggy. If disabled, cross-valiation is only run on save'
    )
   
    # apply the current model with the user defined parameters
    parameters = project.create_model_params(model_name, r, sill, nugget, shape, cross_validate=always_cv)
    x, y = project.apply_variogram_model(parameters)

    # Main plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,0], mode='lines', line=dict(color='gray'), fill=None, name='lower bound')
    )
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,1], mode='lines', line=dict(color='gray'), fill='tonexty', name='upper bound')
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', line=dict(color='green', width=2), name=f'{model_name.capitalize()} model')
    )

    fig.update_layout(
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01
        )
    )

    st.markdown(r"""
    ## Experimental variogram error margin

    The follow plot shows the $\frac{%d}{256}$ 
    observation uncertainty level error margins for the 
    experimental variogram.

    """ % sigma)
    st.plotly_chart(fig, use_container_width=True)

    # if not stopped load all models at current specs to get some statistics
    all_mods = project.load_model_params(sigma=sigma, model_name=model_name)
    compare_fit = np.mean([p['fit'] for p in all_mods])
    compare_rmse = np.mean([p['rmse'] for p in all_mods])

    # output the parameters as json
    success_container = st.empty()
    detail_expander = st.beta_expander('DETAILS', False)
    left, right = detail_expander.beta_columns((3, 7))
    left.markdown(r"""
    ### Current parameterization
    """)
    left.json(parameters)


    # show fit as plot
    fit_chart = go.Figure()
    fit_chart.add_trace(go.Indicator(
        mode='number+gauge+delta',
        gauge=dict( axis=dict(range=[0, 1], tickformat='%')),
        delta=dict(reference=compare_fit / 100),
        value=parameters.get('fit') / 100,
        domain=dict(x=[0.0, 0.45], y=[0., 1.]),
        title='fit'
    ))
    fit_chart.add_trace(go.Indicator(
        mode='number+delta',
        gauge=dict( axis=dict(range=[0, 1], tickformat='%')),
        delta=dict(reference=compare_rmse),
        value=parameters.get('rmse'),
        domain=dict(x=[0.5, 0.95], y=[0., 1.]),
        title='RMSE'
    ))
    fit_chart.update_layout(
        margin=dict(t=5, b=5, l=30, r=30)
    )
    right.plotly_chart(fit_chart, use_container_width=True)

    # add a button to save the Parameters
    save_requested = left.button('SAVE MODEL', key='save1')
    save_requested_sidebar = st.sidebar.button('SAVE MODEL', key='save2')

    # handle model saving
    if save_requested or save_requested_sidebar:
        try:
            with st.spinner('Cross-validating and hashing...'):
                parameters = project.create_model_params(model_name, r, sill, nugget, shape, cross_validate=True)
                project.save_model_params(parameters)
                success_container.success(f"Model parameters saved!\nMD5-checksum: {parameters.get('md5')}")

        except AttributeError:
            st.warning('These parameters are already saved to the project')

    else:
        st.info(f"The project has {len(all_mods)} {model_name.capitalize()} models for {sigma}/{256} uncertainty level. Adjsut the parameters in the sidebar to find more good models.")


    return project

if __name__ == '__main__':
    st_app()