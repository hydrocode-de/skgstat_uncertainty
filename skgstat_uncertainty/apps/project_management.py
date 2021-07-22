from typing import Callable, List, Dict
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from skgstat_uncertainty.core import Project
from skgstat_uncertainty import components


#@st.cache
def variogram_model_plots(vario_func: Callable[..., tuple], bins: np.ndarray, error_bounds: np.ndarray, models: List[dict], sigma: int = None, container=None):
    if container is None:
        container = st
    
    # filter models for sigma level
    models = [m for m in models if m['sigma_obs'] == sigma]

    # create the plotting area
    opts, plot = container.beta_columns((3,7))

    opts.markdown('Select one or more models to plot the model')
    # create the options
    used_idx = opts.multiselect(
        'Add models to the Plot',
        options=list(range(len(models))),
        format_func=lambda i: f"<ID={models[i]['id']}> {models[i]['model'].capitalize()} model",
        key=f'model_select_{sigma}'
    )
    
    # filter by used index
    used_models = [models[i] for i in range(len(models)) if i in used_idx]
    
    colors = []
    for m in used_models:
        col = opts.color_picker(
            f"<ID={m['id']}> color",
            value='#35D62E',
            key=f"col_{sigma}_{m['id']}"
        )
        colors.append(col)


    if len(used_models) == 0:
        return

    # create the figure
    fig = go.Figure()
    
    # add the error bounds
    fig.add_trace(
        go.Scatter(x=bins, y=error_bounds[:,0], mode='lines', line=dict(color='grey'), fill='none', name='lower bound')
    )
    fig.add_trace(
        go.Scatter(x=bins, y=error_bounds[:,1], mode='lines', line=dict(color='grey'), fill='tonexty', name='upper bound')
    )
    
    # apply the models
    for model, col in zip(used_models, colors):
        x, y = vario_func(model)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(width=1.25, color=col),
                name=f"<ID={model['id']}> {model['model'].capitalize()} model",
            )
        )
    
    # update layout
    fig.update_layout(
        title=f"Variogram Models fitted at {sigma} uncertainty level",
        legend=dict(orientation='h')
    )

    plot.plotly_chart(fig, use_container_width=True)
    return fig


def st_app(project: Project):
    if project is None:
        project = Project()

    st.title('Data and results')
    st.markdown('### Project files')

    # edit mode
    edit = st.sidebar.checkbox('edit')
    export = st.sidebar.checkbox('show export options', value=True)

    # add plot controls
    plot = st.sidebar.checkbox('Show result plots', value=False, help="It's recommended to only activate 'export' or 'plot'")
    
    # README
    readme_expander = st.beta_expander('README.md', expanded=True)
    # read the README
    with open(project.path + '/README.md') as f:
        readme = f.read()
    
    # handle edit
    if edit:
        readme_msg = readme_expander.empty()
        readme_edit_form = readme_expander.form('readme-edit')
        new_readme = readme_edit_form.text_area('README', value=readme, height=350)
        readme_saved = readme_edit_form.form_submit_button('Save changes')

        if readme_saved:
            with open(project.path + '/README.md', 'w') as f:
                f.write(new_readme)
            readme_msg.success('File saved!')
    else:
        readme_expander.markdown(readme)
    
    # get the bibtex
    bib_expander = st.beta_expander('bibliography.bib', expanded=False)
    bib_expander.markdown("If you use the data generated with this application, you **have to** cite at least the following publications:")
    
    # read
    with open(project.path + '/bibliography.bib') as f:
        bib = f.read()
    
    if edit:
        bib_msg = bib_expander.empty()
        bib_edit_form = bib_expander.form('bib-edit')
        new_bib = bib_edit_form.text_area('BIBLIOGRAHY', value=bib, height=450)
        bib_saved = bib_edit_form.form_submit_button('Save changes')

        if bib_saved:
            with open(project.path + '/bibliography.bib', 'w') as f:
                f.write(new_bib)
            bib_msg.success('File saved!')

    bib_expander.code(bib, language='bibtex')


    st.markdown(f"""
    ### Variograms

    The tables below list all cached intermediate data and results
    that are stored in the Project `{project.name}`. Every base variogram
    is contained in its own expander 
    """)

    # go for each variogram
    for v_idx, v_dict in enumerate(project.config().get('variograms')):
        
        # Simulations
        with st.spinner('Crunching some numbers...'):
            # set the current variogram
            md5 = v_dict['md5']
            project.vario = md5

            # create the container
            expander = st.beta_expander(f"Variogram: {v_dict['name']}", expanded=v_idx == 0)
            
            # show a spinner while building the Table
            levels = project.config()['sigma_levels'].get(md5, {})
            level_table = []
            for N, level in levels.items():
                # now it's getting wild
                level_table.extend([{'Simulations': int(N.split('_')[0]), **{k:v for k,v in d.items() if k != 'bounds'}} for d in level])

            # Model fits
            all_params = project.load_model_params()

            # krigings
            kriged_fields = project.kriged_fields_info(0, 100)

            expander.markdown('### Overview')
            expander.text(f"Table 1: Overview")
            expander.table([
                {'Label': 'Uncertainty simulations', 'Amount': len(level_table)},
                {'Label': 'Fitted Models', 'Amount': len(all_params)},
                {'Label': 'Kriged fields', 'Amount': len(kriged_fields)}
            ])

            expander.markdown('### Monte-Carlo Simulations')
            expander.text(f"Table 2: Experimental variogram uncertainty levels simulated for {v_dict['name']}")
            expander.table(level_table)
            if export:
                components.table_export_options(pd.DataFrame(level_table), container=expander, key=f'levels{v_idx}')
        
            expander.markdown("### Theoretical Variogram models")
            # Variogram Plot
            if plot:
                expander.markdown("Select a Monte-Carlo Simulation from the Dropdown below to plot the different models fitted to this simulation result, as listed in the Table below.")
                sigma_lvls = project.get_error_levels(as_dict=True)
                sigmas = expander.multiselect(
                    'Simulation background data', 
                    options=project.get_error_levels(),
                    format_func=lambda l: sigma_lvls[l] 
                )
                
                for sigma in sigmas:
                    variogram_model_plots(
                        vario_func=project.apply_variogram_model,
                        bins=project.vario.bins,
                        error_bounds=project.load_error_bounds(5),
                        models=all_params,
                        sigma=sigma,
                        container=expander
                    )

            expander.text(f"Table 3: Theoretical variogram models fitted within experimental base data of {v_dict['name']}")
            expander.table(all_params)
            if export:
                components.table_export_options(pd.DataFrame(all_params), container=expander, key=f'params{v_idx}')

            expander.markdown('### Kriging Results')
            if plot:
                expander.markdown('Select any of the kriged result fields to inspect the individual interpolations')

                field_idx = expander.multiselect(
                    'Select interpolation results',
                    options=[i for i in range(len(kriged_fields))],
                    format_func=lambda i: f"<ID={kriged_fields[i]['id']}> {kriged_fields[i]['model'].capitalize()} model"
                )
                # filter fileds
                plot_kriged = [kriged_fields[i] for i in range(len(kriged_fields)) if i in field_idx]
                
                components.detailed_kriged_plot(
                    field_load_func=project.load_single_kriging_field,
                    params=plot_kriged,
                    container=expander,
                    obs=project.vario_plot_obs
                )


            expander.text(f"Table 4: Summary of all interpolations run based on the fitted theoretical models")
            expander.table(kriged_fields)
            if export:
                components.table_export_options(pd.DataFrame(kriged_fields), container=expander, key=f'kriged{v_idx}')

            # maybe remove this again
            expander.markdown('## Saved result files')
            result_files = project.current_results()
            
            if len(result_files) > 0:
                for fname in result_files:
                    expander.write(fname)
            else:
                expander.info('No result charts or tables found.')

if __name__ == '__main__':
    st_app(None)