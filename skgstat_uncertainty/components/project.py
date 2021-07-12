import streamlit as st
import hashlib
import os
import io
import base64

from skgstat_uncertainty.core import Project
#from ..core import Project

def project_management(container=None) -> Project:
    """
    Builds an project management widget that returns the 
    loaded Project instance
    """
    if container is None:
        container = st
    
    # checkout action
    actions = {0: 'Upload a Project', 1: 'Create new Project', 2: 'Select existing Project'}
    container.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    action = container.radio(
        '',
        options=[0, 1],
        format_func=lambda o: actions.get(o),
        index=0
    )

    # UPLOAD
    project_name = None
    if action == 0:
        # add a project file upload
        uploaded_project_file = container.file_uploader(
            'Upload a local project file',
            type='tar.gz',
            accept_multiple_files=False
        )

        # check
        if uploaded_project_file is not None:
            # try to infer project name
            try:
                project_name = os.path.basename(uploaded_project_file.name).split('.')[0]
            except Exception:
                project_name = hashlib.md5(uploaded_project_file.read()).hexdigest()
                uploaded_project_file.seek(0)

            # user uploaded a file
            with st.spinner('Extracting...'):
                buffer = io.BytesIO
                Project.open(uploaded_project_file, extract_path=None, project_name=project_name)
        else:
            project_name = None

    # CREATE
    elif action == 1:
        with container.form('CREATE NEW PROJECT'):
            st.markdown("Create a new Project")
            add_sample = st.checkbox('Add Example: pancake variogram')
            name = st.text_input('Project Name')
            submitted = st.form_submit_button('CREATE')
            
            # check if submitted
            if submitted:
                Project.create(path=os.path.join(os.path.dirname(__file__), '..', 'projects'), add_sample=add_sample, name=name)
                project_name = name

    # SELECT
#    else:
    # build the select
    available_projects = Project.list_extracted_projects()
    
    # build the project select
    project_name = container.selectbox(
        'Project',
        options=['No Project selected', *list(available_projects.keys())],
        index=list(available_projects.keys()).index(project_name) + 1 if project_name is not None else 0
    )

    # load the correct path
    if project_name == 'No Project selected':
        project_path = None
    else:
        project_path = available_projects[project_name]

    # initialze the project
    project = Project(path=project_path)
    
    if project.initialized:
        do_download = container.button(f'DOWNLOAD PROJECT FILES')

        if do_download:
            with st.spinner('Creating Archive...'):
                buffer = io.BytesIO()
                
                # write into the buffer
                buffer = project.save(buffer)

            with st.spinner('Creating download...'):
                b64 = base64.b64encode(buffer.read()).decode()
                href = f"data:file/tar;base64,{b64}"

                # download link
                container.write(f"""
                <a href="{href}" download="{project.name}.tar.gz">Archive finished. Click to download before you click something else. May take a few seconds.</a>
                """, unsafe_allow_html=True)

    # handle project settings
    return project


def variogram_selector(project: Project, container=None) -> Project:
    if container is None:
        container = st
    
    # read all available variogram
    available_varios = project.config().get('variograms')

    vario_md5 = container.selectbox(
        'Select the variogram',
        options=[v['md5'] for v in available_varios],
        format_func=lambda md5: [v for v in available_varios if v['md5'] == md5][0]['name'],
        index=0
    )

    # set the variogram
    project.vario = vario_md5

    return project


def simulation_family_selector(project: Project, container=None) -> Project:
    if container is None:
        container = st
    
    # get the levels from the current config
    vario_config = project.config()['sigma_levels'].get(project._vario, {})

    # make nice
    levels = dict()
    for key in vario_config.keys():
        chunks = key.split('_')
        levels[chunks[0]] = f"{chunks[0]} {chunks[1].capitalize()}"

    # build the selector
    n_level = container.selectbox(
        'Select the Simulation family',
        options=list(levels.keys()),
        format_func= lambda o: levels.get(o)
    )
    
    # update the project
    project.n_iterations = int(n_level)

    return project


if __name__ == '__main__':
    project_management()