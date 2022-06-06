import streamlit as st

from skgstat_uncertainty.api import API
from skgstat_uncertainty.db import DATAPATH, DBNAME
from skgstat_uncertainty.chapters.data_manage import main_app as data_manager
from skgstat_uncertainty.chapters.variogram import main_app as variogram_app
from skgstat_uncertainty.chapters.model_fitting import main_app as fitting_app
from skgstat_uncertainty.chapters.kriging import main_app as kriging_app
from skgstat_uncertainty.chapters.model_simulation import main_app as simulation_app
from skgstat_uncertainty.chapters.model_compare import main_app as compare_app


NAV_NAMES = {
    'data': 'Data Manager',
    'variogram': 'Variogram Estimation',
    'model': 'Theorectical Model Parametrization',
    'kriging': 'Model application - Kriging',
    'simulation': 'Model application - Simulation',
    'compare': 'Results - Compare Kriging'
}
def navigation(names: dict, container = st) -> str:
    # build the navigation
    page = container.selectbox(
        label='Navigation',
        options=list(names.keys()),
        format_func=lambda k: names.get(k)
    )

    return page


def main_app(data_path=DATAPATH, db_name=DBNAME, **kwargs):
    # filter names
    filt_names = kwargs.get('filter_names', [])
    NAMES = {k: v for k, v in NAV_NAMES.items() if k not in filt_names}


    st.set_page_config('SciKit-GStat Uncertainty', layout='wide')
    navigation_expander = st.sidebar.expander('Navigation', expanded=True)

    # navigation
    page_name = navigation(names=NAMES, container=navigation_expander)

    # DEV decide if there are pages, that do not need a API
    api = API(data_path=data_path, db_name=db_name)

    if page_name == 'data':
        data_manager(api=api)
    elif page_name == 'variogram':
        variogram_app(api=api)
    elif page_name == 'model':
        fitting_app(api=api)
    elif page_name == 'kriging':
        kriging_app(api=api)
    elif page_name == 'simulation':
        simulation_app(api=api)
    elif page_name == 'compare':
        compare_app(api=api)
    else:
        st.error('404')


if __name__ == '__main__':
    import fire
    fire.Fire(main_app)
    #main_app()
