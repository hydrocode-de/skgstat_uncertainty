import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import skgstat as skg
from time import time
import numpy as np
import pandas as pd

from skgstat_uncertainty.core import Project
from skgstat_uncertainty import components
#from ..core import Project
#from .. import components


def create_variogram_app(project: Project, save_results=False) -> Project:
    # if there is no project passed, use the default one
    if project is None:
        project = Project()

    # Include some logic to determine data creation
    st.markdown("""
    ## Estimate a new variogram
    Select the checkbox to enable variogram estimation
    """)
    activate_estimate = st.checkbox(
        'Enable variogram estimation',
        value=False
    )

    # check if select or estimate variogram
    if activate_estimate:
        st.info('Here, a dataset upload is missing')

        # add a binning method
        BINNING_METHODS = dict(
            even='Evenly spaced bins',
            uniform='Evenly sized bins',
            kmeans='K-Means clustered centers',
            ward="Ward's hierachical clustering centers",
            sturges="Sturge's rule",
            scott="Scott's rule",
            fd="Freedman-Diaconis estimator",
            sqrt="Squareroot rule",
            doane="Doane's rule",
            entropy="Stable entropy bins"
        )

        # lags are only specified for some methods
        bin_method = st.sidebar.selectbox(
            'Select binning method',
            options=list(range(len(BINNING_METHODS.keys()))),
            format_func=lambda i: list(BINNING_METHODS.values())[i],
            index=0
        )
        bin_method = list(BINNING_METHODS.keys())[bin_method]

        if bin_method in ('even', 'uniform', 'kmeans', 'ward', 'entropy'):
            # add the controls to the sidebar
            n_lags = st.sidebar.slider(
                'number of lag classes',
                min_value=5,
                max_value=100,
                value=20
            )
        else:
            n_lags=20
        
        # maxlag
        maxlag = st.sidebar.slider(
            'Maximum lag distance',
            min_value=15,
            max_value=1000,
            value=500,
            step=10,
            help='Maxlag help whatsoever'
        )
        # some sampling settings
        st.sidebar.markdown('## Sampling')
        N = st.sidebar.slider(
            'Sample Size',
            min_value=20,
            max_value=1500,
            value=150,
            step=10
        )
        use_seed = st.sidebar.checkbox('Seed random sampling', value=True, help='Same seeded samplings will always yield the same sample')
        if use_seed:
            seed = st.sidebar.number_input('Seed', min_value=0, max_value=99999999, step=127, value=42)
        else:
            seed = None

        # estimate
        vario = project.create_base_variogram(N=N, seed=seed, bin_func=bin_method, n_lags=n_lags, maxlag=maxlag)

        # create the graph
        vario_chart = go.Figure(go.Scatter(
            x=vario.bins,
            y=vario.experimental,
            mode='markers',
            marker=dict(size=8)
        ))
        vario_chart.update_layout(
            title='Experimental base variogram',
            xaxis=dict(title='Lag [-]'),
            yaxis=dict(title=f'Semi-variance [{vario.estimator.__name__}]')
        )

        # show plot
        st.plotly_chart(vario_chart, use_container_width=True)

        # add fields to store
        with st.form('desc'):
            title = st.text_input('Dataset Title', help='Use an expressive name to identify your Dataset later again')
            description = st.text_area('Description', value="My dataset", help="Give a short description what your variogram is about")
            submitted = st.form_submit_button('Save')
        
        if submitted:
            # load the original
            # TODO: this needs to be dynamic
            pan = skg.data.pancake_field().get('sample')
            vario_md5 = project.save_base_variogram(vario, pan, title, description)

            # set the proeject to the current md5
            project.vario = vario_md5

            # save 
            if save_results:
                vario_chart.write_image(project.result_path + f'/{vario_md5}_base_variogram.pdf')

            st.success(f'{title} saved!')
            # do the saving then return
            return project
        else:
            st.warning('Datset not saved yet')

        # stop further execution until finished
        st.stop()

    # later
    # DATA_OPTIONS = {info['md5']: info.get('name', info['md5']) for info in project.config().get('variograms', [])}
    # vario_md5 = st.selectbox(
    #     'Select a saved variogram',
    #     options=list(DATA_OPTIONS.keys()),
    #     format_func=lambda md5: DATA_OPTIONS.get(md5)
    # )
    project = components.variogram_selector(project, st.sidebar)

    # set the selected variogram 
    if project._vario is None:
        st.info('You need to select a dataset first')
        st.stop()
    
    # return current project
    return project


def create_monte_carlo_app(project: Project, save_results=False) -> Project:
    st.markdown("""
    ## MonteCarlo Simulation

    Run a Monte-Carlo simulation or the selected 
    experimental variogram. The selected level of observation 
    uncertainty will be propagated into the experimental variogram.
    """)

    # add controls
    N = st.sidebar.number_input(
        'Iterations',
        min_value=100,
        max_value=100000,
        value=50000,
        step=1000,
        help='Iterations for each Simulation. 50000 is the recommended number'
    )
    st.sidebar.markdown(r'Set uncertainty level $\sigma$ as $\frac{\sigma}{256}$')
    level = st.sidebar.number_input(
        'observation uncertainty level',
        min_value=0,
        max_value=256,
        value=5,
        step=1
    )
    
    # update project
    project.sigma = level
    project.n_iterations = N

    use_seed = st.sidebar.checkbox(
        'Seed the simulation',
        value=True,
        help='Two simulations of same parameters and seed will produce the same output'
    )
    if use_seed:
        seed = st.sidebar.number_input('Random seed', min_value=0, max_value=99999999, step=1312, value=42)
    else:
        seed = None

    # add an empty container for the result table
    result_table_expander = st.beta_expander(f'Simulations saved at {N} iterations')
    result_table_container = result_table_expander.empty()
    result_table = project.monte_carlo_result_table()
    result_table_container.table(result_table)

    # some plotting tools 
    # progress_tools: st = st.sidebar.beta_expander('PROGRESS SETTINGS')
    show_progress = st.sidebar.checkbox(
        'Plot simulation progress', 
        value=True, 
        help="You can view the error bounds changes during the simulation at the cost of longer runtimes"
    )
    if show_progress:
        plot_interval = st.sidebar.number_input(
            'Plot every x steps',
            min_value=1,
            max_value=1000,
            step=1,
            value=100
        )

    #add start buttons
    do_run = st.button('START SIMULATION')
    do_run2 = st.sidebar.button('START')

    # give a short info
    if not do_run and not do_run2:
        it_sec = project.vario.n_lags * 9.0
        st.info(f'At approx. {it_sec} iterations per second, running {N} iterations needs ~ {round(N / it_sec)} seconds.')
        # st.stop()
    else:
        t1 = time()
        with st.spinner('Running...'):            
            # create the plot space
            if show_progress:
                #convergence_plot = st.line_chart(np.zeros((1, len(project.vario.bins))))
                data = pd.DataFrame(columns=list(range(len(project.vario.bins))), index=pd.RangeIndex(0))
                left_plot, right_plot = st.beta_columns(2)
                #convergence_plot = left_plot.line_chart(data)
                convergence_plot = left_plot.empty()
                bins_plot = right_plot.empty()
                x = project.vario.bins

            # get a progress bar
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            # do the simulation
            for i in project.monte_carlo_simulation(N, level, seed):
                progress_bar.progress(i / N)
                progress_text.text(f'{round((i / N) * 100)}% - {round(time() - t1)} seconds elapsed')

                if not show_progress:
                    continue

                if i % plot_interval == 0:
                    new_data = (np.nanmax(project._mc_output_data, axis=1) - np.nanmin(project._mc_output_data, axis=1))#.reshape(1, -1)
                    data.loc[i] = new_data
                    #convergence_plot.add_rows(data.loc[(i * plot_interval)])
                    conf_fig = px.line(data)
                    conf_fig.update_traces(line=dict(color='darkgreen'))
                    conf_fig.update_layout(showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), title='error margin bounds')
                    convergence_plot.plotly_chart(conf_fig, use_container_width=True)
                    # convergence_plot.line_chart(data)
                    exp_fig = go.Figure()
                    exp_fig.add_trace(go.Scatter(x=x, y=np.nanmin(project._mc_output_data, axis=1), mode='lines', line=dict(color='grey'), fill=None))
                    exp_fig.add_trace(go.Scatter(x=x, y=np.nanmax(project._mc_output_data, axis=1), mode='lines', line=dict(color='grey'), fill='tonexty'))
                    exp_fig.update_layout(showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), title='current experimental varioram')
                    bins_plot.plotly_chart(exp_fig, use_container_width=True)
                
            
            # reset status information
            progress_bar.empty()
            progress_text.empty()

            # save
            project.save_error_bounds(level)

            # save the figures
            if save_results:
                conf_fig.write_image(project.result_base_name % 'convergence_plot.pdf')
                exp_fig.write_image(project.result_base_name % 'experimental_bounds.pdf')
            t2 = time()

        # finished!
        st.success(f'Done after {round(t2 - t1)} seconds.')

    # build the result table, containing this simulation
    result_table = project.monte_carlo_result_table()
    result_table_container.table(result_table)
    components.table_export_options(result_table, container=result_table_expander, key='mc_result1')

    # save if needed
    if save_results:
        with open(project.result_path + f'/{project._vario}_{N}_simulation_overview.tex', 'w') as fs:
            fs.write(result_table.to_latex(index=None))

    return project
    

def st_app(project: Project = None) -> Project:
    st.title('Estimate uncertain variograms')
    st.markdown("""
    This short application let's you estimate variograms
    for a given dataset and then estimate error margins
    for the variogram.

    These error margins can be used in the next chapter
    to find theoretical models within the error margins and
    finally analyse the model choice impact on a kriging
    interpolation in chapter 3.
    """)
    # add the save indicator
    save_results = st.sidebar.checkbox('Save results', True)
    st.sidebar.title('Parameters')

    # first run estimation
    project = create_variogram_app(project, save_results=save_results)

    # then run simulation
    project = create_monte_carlo_app(project, save_results=save_results)

    return project


if __name__ == '__main__':
    st_app()
