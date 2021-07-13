import os
import re
import importlib
from typing import List
import streamlit as st

from skgstat_uncertainty.core import Project
from skgstat_uncertainty import  components


def get_pages(file_names: List[str], chapters=False) -> List[dict]:
    basepath = os.path.dirname(__file__)
    pages = list()

    # compile the pattern
    pat = re.compile('st\.title\(\'(.+?)\'\)')
   
    # TODO replace by glob later
    for idx, fname in enumerate(file_names):
        base = os.path.splitext(os.path.basename(fname))[0]

        # get the title
        with open(os.path.join(basepath, fname), 'r') as f:
            code = f.read()

            # find title
            r = pat.search(code)
            if r is not None:
                title = r.group(1)
            else:
                title = ' '.join([c.capitalize() for c in base.split('_')])
            
            # build title
            if chapters:
                title = f"Chapter {idx + 1} - {title}"

        # append
        #mod = importlib.import_module(base, 'skgstat_uncertainty.apps')
        mod = importlib.import_module('skgstat_uncertainty.apps')
        app = getattr(mod, base)
        # mod = importlib.import_module(f'..{base}', __name__)
        # from skgstat_uncertainty import apps
        # mod = getattr(apps, base)
        # app = getattr(mod, 'st_app')
        pages.append(dict(title=title, app=app))

    return pages



# create navigation, this could be read from the project
page_list = [
    *get_pages(['home.py'], False),
    *get_pages(['variogram_uncertainty.py', 'model_fit.py', 'model_compare.py'], True),
    *get_pages(['project_management.py'], False)
]
PAGES = {i: p for i, p in enumerate(page_list)}


def main_app():
    st.set_page_config(layout='wide', page_title='SciKit-GStat uncertainty extension')

    # create the options expander
    options = st.beta_expander('OPTIONS', expanded=True)

    # put the header here
    nav_item = options.selectbox(
        'Navigation',
        options=list(PAGES.keys()),
        format_func=lambda idx: PAGES.get(idx, {}).get('title', f'Page {idx + 1}')
    )

    # Inject Project management app
    project = components.project_management(container=options)

    # for index 0 just skipt Project Management
    if nav_item != 0:
        # error and halt if the project is not initialized
        if not project.initialized:
            st.info('You need to select, create or upload a project file (.tar.gz) first')
            st.stop()

    # load the correct app
    page = PAGES.get(nav_item)

    # page config
    app = page['app']
    project = app(project)


if __name__ == '__main__':
    main_app()
