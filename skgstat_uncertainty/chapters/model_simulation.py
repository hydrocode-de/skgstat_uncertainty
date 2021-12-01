import streamlit as st
import numpy as np
import gstools as gs

from skgstat_uncertainty.api import API
from skgstat_uncertainty import components

def main_app(api: API):
    st.title('Model Simulation')
    st.markdown("This chapter is an alternative to the Kriging chapter. Instead of predicting the target field, it will run geostatistical simulations.")

    # load the dataset and interval to be used
    dataset, vario, interval = components.data_selector(api=api, container=st.sidebar)
    models = interval.models

    # create a table anchor
    table_anchor = components.model_table(models=models, variant='dataframe')

    # ----> DEV, use only one model
    model = models[0]
    variogram = model.variogram
    
    # covert to gstools
    gs_model = variogram.to_gstools()
    # initialize a Kriging instance to condition the SRF
    krige = variogram.to_gs_krige()
    cond_srf = gs.CondSRF(krige)

    # create a grid
    x = y = range(100)

    # <---- DEV

    # some elements
    sim_exp = st.sidebar.expander('SIMULATION SETTINGS', expanded=True)
    n = sim_exp.number_input('Number of simulations', value=10, min_value=1, max_value=100)
    start = sim_exp.button('START simulation')
    
    if start:
        # remove table anchor
        table_anchor.empty()

        # show spinner
        fields = []
        with st.spinner('Simulating...'):
            progress_bar = st.progress(0)
            for i in  range(n):
                f = cond_srf.structured((x, y))
                fields.append(f)
                
                progress_bar.progress((i + 1) / n)            
            progress_bar.empty()
            field_stack = np.stack(fields, axis=2)
            
            st.stop()            
    else:
        st.stop()


if __name__ == "__main__":
    st.set_page_config(page_title='Model Simulation', layout='wide')
    def run(data_path = None, db_name = None):
        api = API(data_path=data_path, db_name=db_name)
        main_app(api)
    import fire
    fire.Fire(run)
