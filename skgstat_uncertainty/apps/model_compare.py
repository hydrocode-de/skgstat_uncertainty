from typing import Callable, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from time import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skgstat_uncertainty.core import Project
from skgstat_uncertainty import components
#from ..core import Project
#from .. import components

# cache some charts that will not change very often
@st.cache
def variogram_compare_chart(vario_func: Callable, bins: np.ndarray, error_bounds: np.ndarray, all_models: list, excluded_ids: list, as_columns=True) -> go.Figure:
    # create the main figure
    fig = make_subplots(
        rows=1 if as_columns else 2, 
        cols=2 if as_columns else 2, 
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=['Excluded Models', 'Used Models']
    )

    # line up the data
    used_models = [p for p in all_models if p['id'] not in excluded_ids]
    excl_models = [p for p in all_models if p['id'] in excluded_ids]

    # order
    if as_columns:
        rows = (1, 1)
        cols = (1, 2)
    else:
        rows = (1, 2)
        cols = (1, 1)


    for row, col, models in zip(rows, cols, (excl_models, used_models)):

        # plot error bounds
        fig.add_trace(
            go.Scatter(x=bins, y=error_bounds[:,0], mode='lines', line=dict(color='grey'), fill='none', name='lower bound'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=bins, y=error_bounds[:,1], mode='lines', line=dict(color='grey'), fill='tonexty', name='upper bound'),
            row=row, col=col
        )

        # add the models
        for model in models:
            # evaluate
            x, y = vario_func(model)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(width=1.2, color='darkgreen'),
                    name=f"<ID={model['id']}> {model['model'].capitalize()} model"),
                row=row, col=col
            )
        
        # some layout
        fig.update_layout(
            legend=dict(orientation='h')
        )

    # return
    return fig


@st.cache(allow_output_mutation=True)
def main_result_charts(mean: np.ndarray, upper: np.ndarray, lower: np.ndarray, lo: int, hi: int, n_fields: int, obs: Tuple[np.ndarray] = None) -> Tuple[go.Figure]:
    # define only one layout
    layout = dict(
        yaxis=dict(scaleanchor='x'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )

    # check obs
    if obs is not None:
        obs_x, obs_y, vals = obs
    
    # build the charts
    # confidence interval
    conf_chart = go.Figure(go.Heatmap(z = (upper - lower).T, colorscale='Hot'))
    conf_chart.update_layout(**layout, title=f'{lo}% - {hi}% confidence interval range')

    # mean
    mean_chart = go.Figure(go.Heatmap(z=mean.T, colorscale='Earth'))
    mean_chart.update_layout(**layout, title=f'Mean of {n_fields} fields')

    if obs is not None:
        obs_scatter = go.Scatter(
            x=obs_x,
            y=obs_y,
            mode='markers',
            marker=dict(color='white', size=4),
            name='Observations',
            text=[str(_) for _ in vals]
        )
        conf_chart.add_trace(obs_scatter)
        mean_chart.add_trace(obs_scatter)

    return conf_chart, mean_chart

@st.cache
def more_result_charts(mean: np.ndarray, original: np.ndarray = None, hist_cum=False) -> go.Figure:
    fig = go.Figure(go.Histogram(x=mean.flatten(), histnorm='probability density', cumulative_enabled=hist_cum, name='Interpolation'))

    if original is not None:
        fig.add_trace(go.Histogram(x=original.flatten(), histnorm='probability density', cumulative_enabled=hist_cum, name='Original field'))

    fig.update_layout(
        title='Histogram%s' % ('s' if original is not None else ''),
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.05)
    )
    fig.update_traces(opacity=.6)

    return fig


@st.cache
def entropy_chart(H: np.ndarray, colorscale='Darkmint', **kwargs) -> go.Figure:
    # define only one layout
    layout = dict(
        yaxis=dict(scaleanchor='x'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    layout.update(kwargs)
    
    fig = go.Figure(go.Heatmap(z=H.T, colorscale=colorscale))
    
    fig.update_layout(layout)
    return fig

@st.cache
def entropy_cv_chart(cv: np.ndarray) -> go.Figure:
    # define only one layout
    layout = dict(
        yaxis=dict(scaleanchor='x'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )

    fig = go.Figure()

    # create the slider steps
    N = cv.shape[2]
    steps = []

    # add all cv field    
    for i in range(N):
        # add the trace
        fig.add_trace(go.Heatmap(z=cv.T, colorscale='Purples', visible=i==0))

        # create the slider step
        step = dict(
            method='update',
            args=[{'visible': [False] * N}, {'title': f"Iterpolation {i} added Entropy"}]
        )
    
        # make the ith field visible
        step['args'][0]['visible'][i] = True
        steps.append(step)
    
    # create the slider
    sliders = [dict(active=0, steps=steps)]

    # update layout
    layout.update(sliders=sliders)
    fig.update_layout(layout)

    return fig
        

@st.cache
def kriging_surface_chart(stack: np.ndarray, model_names: list) -> go.Figure:
    fig = go.Figure()

    # add all stacks
    for i in range(stack.shape[2]):
        fig.add_trace(
            go.Surface(z=stack[:, :, i].T, showscale=False, opacity=0.3, colorscale='Earth', name=f'{model_names[i].capitalize()} interpolation')
        )
    
    # update layour
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.05),
        height=600
    )

    return fig


# define the main app
def st_app(project: Project = None) -> Project:
    # start the application
    st.title('Compare Models')
    st.markdown("""
    This short application let's you compare different models
    that were fitted with the previous application. With this
    application, you can compare the different models, create or
    update their interpolation and decide which one to use.
    """)
    # add the save indicator
    save_results = st.sidebar.checkbox('Save results', True)
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
    # update
    project.sigma = sigma

    # get the models
    all_models = project.load_model_params(sigma=sigma)

    # create the filter in the sidebar
    st.sidebar.markdown('## Filter models')
    include_fits = st.sidebar.checkbox(
    'Include models within Margin?',
    value=True,
    help='Always include models completely within error bounds'
    )

    evalf = st.sidebar.selectbox(
        'Evaluation function',
        options=['RMSE', 'cv'],
        help="The variogram models can be evaluated by their experimental variogram fit (RMSE) or a kriging cross-validation MAE (cv)"
    )

    std_level = st.sidebar.number_input(
        'Include models within times standard deviation',
        min_value=0.1,
        max_value=10.,
        step=0.5,
        value=1.5,
        help="Automatic quality control: Any model's RMSE deviating more than this times standard deviation from the sample target RMSE will be excluded."
    )
    target = st.sidebar.selectbox(
        f'Sample {evalf.upper()} target',
        options=['min','mean', 'max'],
        help="Specify to which statistical property of all models the individual models' RMSE should be compared. Usually, min."
    )

    # set new settings
    project.filter_include_fit = include_fits
    project.std_level = std_level
    project.std_target = target
    project.model_evalf = evalf

    # prefilter
    project.prefilter_models()
    prefiltered_models = project.prefiltered_models
    pre_filtered_ids = [p['id'] for p in prefiltered_models]

    # now manually exclude models
    excluded_models = st.sidebar.multiselect(
        'Excluded model parameterizations',
        options=[p['id'] for p in all_models],
        default=[p['id'] for p in all_models if p['id'] not in pre_filtered_ids],
        help='Models outside the given standard deviation are automatically excluded'
    )
    # update project
    project.excluded_model_ids = excluded_models
    
    # finally exclude non-used models
    used_models = [p for p in prefiltered_models if p['id'] not in excluded_models]
    
    # output and style all models
    with st.spinner('Loading your models...'):
        st.markdown(r"""
        ## Filter Models
        At $\frac{%d}{256}$ level uncertainty used, currently %d / %d models 
        are selected for further analysis
        """ % (sigma, len(used_models), len(all_models)))
        all_models_df = pd.DataFrame(all_models)

        # build the compare variograms
        compare_container = st.beta_expander('View all models', expanded=False)

        error_bounds = project.load_error_bounds(sigma=sigma)
        variogram_compare = variogram_compare_chart(
            vario_func=project.apply_variogram_model,
            bins=project.vario.bins,
            error_bounds=error_bounds,
            all_models=all_models,
            excluded_ids=excluded_models
        )
        compare_container.plotly_chart(variogram_compare, use_container_width=True)
    
    # if all are excluded, there is nothing to do
    if len(used_models) == 0:
        st.error('Either no models loaded, or all models are excluded')
        st.stop()
    elif len(used_models) <= 3:
        st.warning('Only 3 or less models are used. It is recommended to estimate more models for a proper analysis.')
    
    # show the models
    st.dataframe(all_models_df.style.apply(lambda r: [f"background: {'#ac1900' if r.id in excluded_models else '#003300'}" for c in r], axis=1))
    
    # add export options
    latex_expander = st.beta_expander('EXPORT OPTIONS', False)
    all_models_df['excluded'] = [p['id'] in excluded_models for p in all_models]    
    components.table_export_options(all_models_df, container=latex_expander, key='exporter1')
    
    # Kriging
    st.sidebar.markdown('## Kriging')
    st.markdown(f"""
    ## Kriging

    The {len(used_models)} models listed above are now used to interpolate
    the observations using each of the selected models for Kriging. 

    The application caches kriging results.
    As kriging can take some time, you need to activate that step manually.
    """)

    run_kriging = st.sidebar.checkbox(
        'Activate kriging',
        value=False
    )

    # calculate needed kriging
    required_krige_fields = [p['md5'] for p in used_models]
    is_cached = [md5 in project.cached_fields for md5 in required_krige_fields]

    # check needed kriging
    # everything is there
    if sum(is_cached) == len(required_krige_fields):
        st.success(f"All {len(used_models)} kriging fields found in cache")
    elif run_kriging:
        # show a message
        with st.spinner(f'Kriging is running...'):
            # go for all uncached models
            missing_fields = project.uncached_fields

            progress_bar = st.progress(0.0)
            progress_text = st.text(f"0/{len(missing_fields)}  -   0%")

            t1 = time()
            for i, missing in enumerate(missing_fields):
                params = project.get_params(missing)
                if params is None:
                    st.error(f"Can't find parameterization {missing}")
                    st.stop()
                
                # do the kriging
                field = project.apply_kriging(params)
                project.save_kriging_field(field, missing)

                # update progress bar
                progress_bar.progress((i + 1) / len(missing_fields))
                progress_text.text(f"{i + 1}/{len(missing_fields)}  -  {round((i + 1) / len(missing_fields) * 100)}%")

            t2 = time()
        
        # reload the cache
        project.load_kriging_cache()

        # remove progressbar
        progress_bar.empty()
        progress_text.empty()

        st.success(f"Kriging finished after {round(t2 - t1, 0)} seconds. [{round((t2 - t1) / len(missing_fields), 1)} / interpolation]")
    else:
        st.info(f"""
        Kriging not activated.

        Currently, {len(used_models)} interpolations are needed, from which
        {sum(is_cached)} are found in the Project. The remaining {len(used_models) - sum(is_cached)}
        interpolations would need approx {(len(used_models) - sum(is_cached)) * 16} seconds.
        """)
        st.stop()
    
    # finall load the cached fields - here they are all calculated
    all_fields = project.cached_fields

    # analyze the fileds
    st.markdown(f"""
    ### Single fields
    You can inspect the {len(all_fields)} fields below and analyze their impact on the overall uncertainty.
    """)
    # controls for the quartils
    (lo, hi) = st.sidebar.slider(
        'Select confidence interval',
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        help=f'{len(all_fields)} values per pixel are estimated. Set their confidence bounds'
    )

    # get the confidence_interval
    # cache the confidence interval function
    @st.cache
    def get_conf_interval(lo, hi, func, excluded):
        # this has to rerun if excluded changes
        return func(lo, hi)
    lower, upper, fields_mean, fields_std, field_count = get_conf_interval(lo, hi, project.kriged_field_conf_interval, excluded_models)
    
    # build the container
    left, right = st.beta_columns(2)
    kriged_expander = st.beta_expander('SINGLE KRIGING FIELDS')
    more_plots = st.beta_expander('More charts...', expanded=False)

    # # build the main charts 
    conf_chart, mean_chart = main_result_charts(fields_mean, upper, lower, lo, hi, len(all_fields), obs=project.vario_plot_obs)
    left.plotly_chart(conf_chart, use_container_width=True)
    right.plotly_chart(mean_chart, use_container_width=True)

    # container for detailed plots
    kriged_info = [p for p in project.kriged_fields_info(lower, upper) if p['sigma_obs'] == sigma]
    kriged_idx = kriged_expander.multiselect(
        'Add single interpolation fields to the plot',
        options=list(range(len(kriged_info))),
        format_func=lambda i: f"<ID={kriged_info[i]['id']}> {kriged_info[i]['model'].capitalize()} model"
    )
    used_kriged_info = [kriged_info[i] for i in kriged_idx]
    components.detailed_kriged_plot(
        field_load_func=project.load_single_kriging_field,
        params=used_kriged_info,
        container=kriged_expander,
        obs=project.vario_plot_obs
    )

    # entropy charts
    st.sidebar.markdown('## Interpolation Entropy')
    calculate_entropy = st.sidebar.checkbox('Activate Entropy calculation', value=False, help="The calculation can take several minutes.")
    H_cv = st.sidebar.checkbox('Cross-validate each field', value=False, help='Get per-interpolation entropy contributions (very costy)')

    if H_cv:
        #cv_num = st.sidebar.slider('Select the field', min_value=1, max_value=len(used_models) + 1, value=1)
        cv_num = st.sidebar.selectbox(
            'Select the field',
            options=[i for i in range(len(used_models))],
            format_func=lambda i: f"#{used_models[i]['id']} - {used_models[i]['model']}"
        )
    
    if calculate_entropy:
        with st.spinner('Calculating Entropy... (can take several minutes)'):
            H, cv, H_max = project.kriged_field_stack_entropy(cross_validate=cv_num if H_cv else None)
            H_chart = entropy_chart(H, title="Total cell-based interpolation Entropy")

            # print out maximum entropy
            more_plots.markdown(f"The shown entropy maps are normalized by the maximum entropy of `{round(H_max, 2)} bit`")

            if cv is None:
                more_plots.plotly_chart(H_chart, use_container_width=True)
            else:
                H_left, H_right = more_plots.beta_columns(2)
                H_left.plotly_chart(H_chart, use_container_width=True)

                Hcv_chart = entropy_chart(cv, colorscale='Purples', title=f"Interpolation #{used_models[cv_num]['id']} Entropy contribution")
                H_right.plotly_chart(Hcv_chart, use_container_width=True)

    # histogram
    hist_container = more_plots.empty()
    hist_cum = more_plots.checkbox('Cummulative Distribution Function', value=False)
    
    # check if a 3D plot should be added
    add_3d_plot = more_plots.checkbox('Inspect single fields? (may be slow)', value=False)
    
    # hist_interp = go.Figure(go.Histogram(x=fields_mean.flatten(), histnorm='probability density', cumulative_enabled=hist_cum, name='Interpolation'))
    original = project.original_field
    
    hist_interp = more_result_charts(fields_mean, original, hist_cum)
    hist_container.plotly_chart(hist_interp, use_container_width=True)

    # add the 3D plot if requested
    if add_3d_plot:
        surface_chart = kriging_surface_chart(project.kriged_field_stack, [p['model'] for p in used_models])
        more_plots.plotly_chart(surface_chart, use_container_width=True)

    # save results
    if save_results:
        variogram_compare.write_image(project.result_base_name % 'model_compare.pdf')
        conf_chart.write_image(project.result_base_name % f'kriging_{lo}_{hi}_conf_interval.pdf')
        mean_chart.write_image(project.result_base_name % f'kriging_{lo}_{hi}_interpolation.pdf')
        # std_chart.write_image(project.result_base_name % f'_kriging_{lo}_{hi}_interp_std.pdf')
        hist_interp.write_image(project.result_base_name % f'_kriging_{lo}_{hi}_historgrams.pdf')
        if calculate_entropy:
            H_chart.write_image(project.result_base_name % 'kriging_entropy.pdf')
        if add_3d_plot:
            surface_chart.write_image(project.result_base_name % '_kriging_surfaces.pdf')

        with open(project.result_base_name % 'all_models.tex', 'w') as fs:
            all_models_df['used'] = ['no' if _id in excluded_models else 'yes' for _id in all_models_df.id]
            fs.write(all_models_df.to_latex())

    # single fields statistics
    st.markdown('### Results')
    stack = project.kriged_field_stack
    st.text(f'''
    Interpolation value range:   [{int(np.min(stack))}, {int(np.max(stack))}]
    Confidence interval width:   [{int(np.min(upper - lower))}, {int(np.max(upper - lower))}]
    ''')

    # build result dataframe
    single_info = pd.DataFrame(project.kriged_fields_info(lower, upper))
    single_info.set_index('id', inplace=True)
    st.table(single_info)

    # add export options
    result_expander = st.beta_expander('OPTIONS', False)
    components.table_export_options(single_info, container=result_expander, key='exporter2')

    return project

if __name__ == '__main__':
    st_app()