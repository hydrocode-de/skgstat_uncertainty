import streamlit as st

from skgstat_uncertainty.core import Project

def st_app(project: Project):
    if project is None:
        project = Project()

    st.title('Data and results')

    # edit mode
    edit = st.sidebar.checkbox('edit')
    readme_expander = st.beta_expander('README.md', expanded=False)
    
    # read the README
    with open(project.path + '/README.md') as f:
        readme = f.read()
    
    if edit:
        edit_form = readme_expander.form('readme-edit')
        new_readme = edit_form.text_area('README', value=readme, height=350)
        saved = edit_form.form_submit_button('Save changes')

        if saved:
            with open(project.path + '/README.md', 'w') as f:
                f.write(new_readme)
    else:
        readme_expander.markdown(readme)
    
    st.markdown(f"""
    The tables below list all cached intermediate data and results
    that are stored in the Project `{project.name}`. 
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
            kriged_fields = project.kriged_fields_info(25, 75)

            expander.text(f"Table 1: Overview")
            expander.table([
                {'Label': 'Uncertainty simulations', 'Amount': len(level_table)},
                {'Label': 'Fitted Models', 'Amount': len(all_params)},
                {'Label': 'Kriged fields', 'Amount': len(kriged_fields)}
            ])

            expander.text(f"Table 2: Experimental variogram uncertainty levels simulated for {v_dict['name']}")
            expander.table(level_table)
        
            expander.text(f"Table 3: Theoretical variogram models fitted within experimental base data of {v_dict['name']}")
            expander.table(all_params)

            expander.text(f"Table 4: ")
            expander.table(kriged_fields)

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