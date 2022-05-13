from typing import Tuple

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skgstat as skg
from streamlit_card_select import card_select

from skgstat_uncertainty.api import API
from skgstat_uncertainty import components
from skgstat_uncertainty.models import DataUpload
from skgstat_uncertainty.components.utils import FIT_METHODS, MODELS, BIN_FUNC, ESTIMATORS, KRIGING_METHODS
from skgstat_uncertainty.chapters.data_manage import _options_from_dataset_names


__story_intro = """
This application will guide you through the estimation of an empirical variogram. 
You will learn about the parameters step by step and learn how to fit a model function. 
If you are already familiar with variogram fitting, you can directly jump into the full fitting interface.
"""

__data_intro = """
First of all, you need to select one of the pre-definded data uploads. If you have access to the 
full Uncertain Geostatistics app by [hydrocode](https://hydrocode.de) (LINK HERE), you can use the data-manage
chapter to upload datasets and fields and create new ones.
Use the dropdown to inspect the datasets

Once you found an exciting dataset, click on *continue* to get started with geostatistics!
"""

__bin_intro = """
To learn about spatial correlation in the dataset, that is not dependent on the location, we
need to calculate separating distances between observations. This way, we can correlate observation similarity
to the distance between them.

A variogram being a statistical property of the sample (2nd moment), we need to aggregate the similarity
in observations at different separating distance lag classes. Therefore, for all combinations of point pairs, 
their distance is calculated and saved as a distance matrix.
This enables us to inspect increasing variablity with increasing distance as a general pattern.

Using Python's `skgstat` library, we can choose between many different methods that will do the binning for us.
Some of them need the number of lag classes as a parameter, others will estimate this automatically.
"""

__est_intro = """
After aggreagting all pairs of observations into distinct lag classes, we need an estimator to express (dis-)similarity
between the members in each group. We use so called semi-variance estimators, which are all largely built on the
univariante variance.
Thus, we are comparing how the varaince in observation residuals changes with increasing distance.

Choose one of the pre-defined estimators and inspect the changes to the overall structure of the empirical variogram.
"""

__fit_method_intro = """
As of now, the empirical variogram is finished. You can already learn a lot from this figure. 
The most important question is if the data actually bears a spatial auto-correlation. Does the semi-variance
increase with distance?

If not, it's not really worth to continue using geostatistics, as anything from here on, will rely on the empirical
variogram and how well it captures the spatial structure of the data.

The empirical variogram is characterized by three different parameters:

  * **sill** is the maximium semi-variance, which is approached asymtotically by the variogram
  * **range** is the distance at which the sill is reached. Beyond this distance, the sample is spatially uncorrelated
  * **nugget** is the y-axis intercept. It's ratio to sill describes the un-explainable share of the sample variance which is nugget + sill

It is generally possible in `skgstat` to turn the use of a nugget effect completely off, by setting the `use_nugget` parameter.
You can estimate the parameters by eye, or capture them more systematically by fitting a theoretical variogram function
to the empirical variogram. This will also let you use the model parameterization in more sophisticated geostatistical applications.
Using `skgstat` you can choose between different methods for fitting.
"""

def _code_sample(**kwargs):
    code = """import skgstat as skg\nfrom skgstat_uncertainty.api import API\n\napi = API()\n"""
    
    # data loaded
    if 'data_id' in kwargs:
        code += f"dataset = api.get_upload_data(id={kwargs['data_id']})\n"
        code += "\n# extract data\ncoords = list(zip(dataset.data['x'], dataset.data['y']))\n"
        code += "values = dataset.data['v']\n"
    else:
        return code
    
    # build the params
    others = ['data_id', 'saw_intro', 'done_variogram', 'kriging_enabled']
    kw = [f"\t{k}={v},\n" for k, v in kwargs.items() if k not in others and isinstance(v, (int, float, bool))]
    kw.extend([f"\t{k}='{v}',\n" for k, v in kwargs.items() if k not in others and isinstance(v, str)])

    # something there?
    if len(kw) > 0:
        code += f"\nvario = skg.Variogram(\n\tcoordinates=coords,\n\tvalues=values,\n{''.join(kw)})\n"
    
    return code


def check_story_mode(api: API):
    """Check if the user wants to be guided through the variogram fit"""
    # if story mode disabled, break out
    if st.session_state.get('saw_intro', False):
        return
    
    # show the dialog
    st.title('Variograms')
    st.markdown(__story_intro)
    l, r = st.columns(2)
    
    # add buttons
    l.markdown('### Guide me through')
    ok_guide = l.button('GO', key='btn_guide')

    r.markdown("### I'm an expert")
    ok_expert = r.button('GO', key='btn_expert')

    if ok_guide:
        st.session_state.saw_intro = True
        st.experimental_rerun()
    elif ok_expert:
        # TODO add all settings here
        st.session_state.saw_intro = True
        st.session_state.bin_method = 'even'
        st.session_state.n_lags = 15
        st.session_state.estimator = 'matheron'
        st.session_state.model = 'spherical'
        st.session_state.fit_method = 'trf'
        st.session_state.use_nugget = False
        st.experimental_rerun()

    else:
        st.stop()


def data_select(api: API) -> DataUpload:
    """Give a full page intro to data"""
    if not st.session_state.get('story_mode', True) or hasattr(st.session_state, 'data_id'):
        # build the data select into the sidebar
        dataset = components.data_selector(api, stop_with='data', container=st.sidebar)
        st.session_state.data_id = dataset.id
        return dataset
    
    # story mode
    st.title('Select a Dataset')
    st.markdown(__data_intro)
    controls = st.columns((8,2))

    # Get the dataset names and convert to card dicts
    all_names = api.get_upload_names(data_type='sample')
    options = _options_from_dataset_names(api, all_names, add_button=False)

    # force to the sidebar
    with st.sidebar.container():
        st.title('Select a Dataset')
        data_id = card_select(options=options, spacing=5)
    
    # if a dataset is selected, show it
    if data_id is None:
        st.warning('No Dataset selected.')
        st.stop()
    else:
        dataset = api.get_upload_data(id=data_id)
    
    # add the plot
    left, right = st.columns((6,4))
    left.markdown('### Dataset plot')
    components.dataset_plot(dataset, disable_download=True, container=left)
    
    # add a data preview
    df = pd.DataFrame({k: v for k, v in dataset.data.items() if k in ('x', 'y', 'v')})
    
    # description
    if 'description' in dataset.data:
        right.markdown('### Dataset description')
        right.markdown(dataset.data['description'])
    
    right.markdown('### Data View')
    right.dataframe(df)

    if 'origin' in dataset.data:
        right.markdown(f"### Origin\n{dataset.data['origin']}")
        
    # add the button
    controls[0].markdown('##### Finished?')
    ok = controls[0].button('CONTINUE')

    if ok:
        st.session_state.data_id = dataset.id
        st.experimental_rerun()
    else:
        st.stop()


def binning(api: API, dataset: DataUpload, expander = st.sidebar) -> None:
    """Learn about binning methods and set the parameters to session state"""
    # base variogram is always needed
    v = dataset.base_variogram()

    # break out
    if not st.session_state.get('story_mode', True) or hasattr(st.session_state, 'bin_method'):
        expander.selectbox('Binning Method', options=list(BIN_FUNC.keys()), format_func=lambda k: BIN_FUNC.get(k), key='bin_method')
        opt = expander.selectbox('Maxlag Method', options=['none', 'median', 'mean', 'absolute'], format_func=lambda k: k.capitalize())
        if opt == 'none':
            opt = None
        if opt == 'absolute':
            expander.slider('Maxlag', key='maxlag', min_value=float(v.bins[0]), max_value=float(np.max(v.distance)), value=float(np.median(v.distance)))
        else:
            st.session_state.maxlag = opt
        if st.session_state.bin_method in ('even', 'uniform', 'kmeans', 'ward'):
            expander.number_input('Number of Lags', min_value=5, max_value=100, key='n_lags')
        return
    
    # story mode
    st.title('Binning')
    st.markdown(__bin_intro)
    btn = st.empty()

    # build the controls
    plot_area = st.empty()
    l, r = st.columns((6, 4))
    l.markdown('### Configure binning')
    _m = l.selectbox('Binning Method', options=list(BIN_FUNC.keys()), format_func=lambda k: BIN_FUNC.get(k))
    opt = l.selectbox('Maxlag Method', ['none', 'median', 'mean', 'absolute'], format_func=lambda k: k.capitalize())
    if opt == 'none':
        opt = None
    if opt == 'absolute':
        _max = l.slider('Maxlag', min_value=float(v.bins[0]), max_value=float(np.max(v.distance)), value=float(np.median(v.distance)))
    else:
        _max = opt
    if _m in ('even', 'uniform', 'kmeans', 'ward'):
        _n = int(l.number_input('Number of Lags', min_value=5, max_value=100, value=15))
    else:
        _n = None

    # code example
    c = _code_sample(data_id=dataset.id, bin_func=_m, maxlag=_max, **{'n_lags': _n for i in range(1) if _n is not None})
    r.markdown('### Python sample code')
    r.code(c + "vario.distance_difference_plot()", language='python')

    # add the graph
    fig = variogram_plot(api, dataset, bin_method=_m, n_lags=_n if _n is not None else 10, maxlag=_max)
    plot_area.plotly_chart(fig, use_container_width=True)

    # add the button
    ok = btn.button('CONTINUE')
    if ok:
        st.session_state.bin_method = _m
        if _n is not None:
            st.session_state.n_lags = _n
        st.experimental_rerun()
    else:
        st.stop()


def estimator(api: API, dataset: DataUpload, expander = st.sidebar) -> None:
    """Learn about SciKit-GStat's implemented estimators and set the parameters to session state"""
    # break out
    if not st.session_state.get('story_mode', True) or hasattr(st.session_state, 'estimator'):
        expander.selectbox('Semivariance estimator', options=list(ESTIMATORS.keys()), format_func=lambda k: ESTIMATORS.get(k), key='estimator')
        return
    
    # story mode
    st.title('Semivariance Estimator')
    st.markdown(__est_intro)
    btn = st.empty()

    # build the main areas
    plot_area = st.empty()
    l, r = st.columns((6, 4))

    # selectbox
    l.markdown('### Select the estimator')
    _est = l.selectbox('Semi-variance estimator', options=list(ESTIMATORS.keys()), format_func=lambda k: ESTIMATORS.get(k))

    # code example
    c = _code_sample(estimator=_est, **st.session_state)
    r.markdown('### Python sample code')
    r.code(c + "# make a plot\nvario.plot()", language='python')

    # add the plot
    fig = variogram_plot(api, dataset, estimator=_est)
    plot_area.plotly_chart(fig, use_container_width=True)

    # add the button
    ok = btn.button('CONTINUE')
    if ok:
        st.session_state.estimator = _est
        st.experimental_rerun()
    else:
        st.stop()


def fit_method(api: API, dataset: DataUpload, expander = st.sidebar) -> None:
    """Learn about SciKit-GStat's implemented fit methods and set the parameters to session state"""
    # break out
    if not st.session_state.get('story_mode', True) or hasattr(st.session_state, 'fit_method'):
        expander.selectbox('Fit Method', options=list(FIT_METHODS.keys()), format_func=lambda k: FIT_METHODS.get(k), key='fit_method')
        expander.checkbox('Use nugget', key='use_nugget')
        return

    # story mode
    st.title('Select a fitting method')
    st.markdown(__fit_method_intro)
    btn = st.empty()

    # selectbox
    _fit = st.selectbox('Fit Method', options=list(FIT_METHODS.keys()), format_func=lambda k: FIT_METHODS.get(k))
    _nug = st.checkbox('Use nugget?', value=False)

    # check out which fitting method will be used
    l, r = st.columns(2)
    if _fit == 'trf':
        l.markdown('### Trust Region Reflective')
        l.markdown("TRF fitting is a least squares approach that minimizes the deviation of calculated to modeled semi-variances. \
            \rIt's a very robust method, that usually converges, but it is also a bit slower than other methods.\
            \rTRF is a **bounded** method, which is constrained by `skgstat` to a valid parameter space.")
        r.info('This will be a preview one day')
    elif _fit == 'lm':
        l.markdown('### Levenberg-Marquardt')
        l.markdown("Levenberg-Marquardt (lm) is least squares aooriach that minimizes the deviation of calculated to modeled semi-variances. \
            \rLM is faster than TRF, but at the cost of stability. As it is **unbounded**, the parameter space is **not** limited \
            \rto a valid parameter space and the method can yield incorrect variogram parameters.")
        r.info('This will be a preview one day')
    elif _fit == 'ml':
        l.markdown('### Parameter maximum likelihood')
        l.markdown("This is a maximum likelihood approach. However, it is **not** maximum likelihood fitting, where the likelihood function\
            \rof the model itself is derived. This appraoch also minimizes the deviation of calculated to modeled semi-variances, like\
            \rin a least squares approach. In each iteration, the parameter distribution function is aadjusted to the current best \
            \rset and the parameters most likely parameterizing the model as a member of this distribution are chosen for the next iteration.")
        r.info('This will be a preview one day')
    elif _fit == 'manual':
        l.markdown('### Manual Fitting')
        l.markdown("Another powerful method is to use the human brain as a optimizer and adjust parameters manually until the model feels right.")
        r.info('This will be a preview one day')
    
    # add the button
    ok = btn.button('CONTINUE')
    if ok:
        st.session_state.fit_method = _fit
        st.session_state.use_nugget = _nug
        st.experimental_rerun()
    else:
        st.stop()


def fitting(api: API, dataset: DataUpload, expander = st.sidebar) -> None:
    """Depending on the fitting method set in session_state, a fitting control panel is shown with appropriate parameters"""
    # fitting has no story mode
    fit_method = st.session_state.fit_method

    # add the models
    expander.selectbox('Theoretical Model', options=list(MODELS.keys()), format_func=lambda k: MODELS.get(k), key='model')
    
    # check what kind of fitting method is selected
    if fit_method == 'manual':
        # get the base variogram
        vario = base_variogram(dataset, fit_method='trf')

        if vario.use_nugget:
            nug, sill = expander.slider('Nugget & Sill', min_value=0.0, max_value=float(np.max(vario.experimental)), value=(0.0, float(np.mean(vario.experimental))))
            st.session_state.nugget = nug
            st.session_state.sill = sill
        else:
            sill = expander.slider('Sill', min_value=0.0, max_value=float(np.max(vario.experimental)), value=float(np.mean(vario.experimental)), key='sill')
        
        # add the range
        expander.slider('Effective range', min_value=float(vario.bins[0]), max_value=float(vario.bins[-1]), value=float(np.mean(vario.bins)), key='range')
    else:
        # handle fit sigma
        SIG = dict(none='Equal Weights', linear='Linear decreasing', exp='Logarithmic decreasing', sq='Exponential decreasing')
        fs = expander.selectbox('Auto-fit weights', options=list(SIG.keys()), format_func=lambda k: SIG.get(k))
        if fs == 'none':
            st.session_state.fit_sigma = None
        else:
            st.session_state.fit_sigma = fs


def variogram(api: API, dataset: DataUpload, always_plot: bool = True) -> None:
    """Show the variogram, along with variogram parameters, code and measures to the user"""
    story_mode = True

    # story mode has to be handled different here
    if not st.session_state.get('story_mode', True) or hasattr(st.session_state, 'done_variogram'):
        if always_plot:
            story_mode = False
        else:
            return
    
    # story mode
    if story_mode:
        st.title('Variogram')
        st.markdown("""Here you can check out your variogram. You can also continue to do some kriging, but that is still beta""")
        btn = st.empty()

    # add the plot
    vario = base_variogram(dataset)
    fig = variogram_plot(api, dataset, variogram=vario)
    st.plotly_chart(fig, use_container_width=True)

    # add measures, parameters and code
    cols = st.columns((2, 2, 5))

    # variogram parameters
    par_exp = cols[2].expander('Variogram parameters', expanded=not story_mode)
    par_exp.table([
        {'Parameter': 'Effective range', 'Value': vario.parameters[0]},
        {'Parameter': 'Sill', 'Value': vario.parameters[1]},
        {'Parameter': 'Nugget', 'Value': vario.parameters[-1]},
        {'Parameter': 'Shape', 'Value': vario.parameters[2] if len(vario.parameters) > 3 else None},
    ])

    # code
    if story_mode:
        c_exp = cols[2].expander('Python sample code', expanded=True)
        c = _code_sample(**st.session_state)
        c += "\n# plot variogram\nvario.plot()"
        c_exp.code(c, language='python')

    # measures
    cols[0].metric('RMSE', vario.rmse.round(1))
    cols[1].metric('Cross-validation', vario.cross_validate().round(1))

    # button
    if story_mode:
        ok = btn.button('CONTINUE')
        if ok:
            st.session_state.done_variogram = True
            st.experimental_rerun()
        else:
            st.stop()


@st.experimental_memo
def _apply_kriging(_dataset: DataUpload, vario_params: dict, grid_resolution: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    # get the variogram
    vario = base_variogram(_dataset, **vario_params)

    # build the grid
    x_range = np.linspace(vario.coordinates[:,0].min(), vario.coordinates[:,0].max(), grid_resolution)
    y_range = np.linspace(vario.coordinates[:,1].min(), vario.coordinates[:,1].max(), grid_resolution)

    # get the kriging instance
    krige = vario.to_gs_krige(**kwargs)

    field, sigma = krige.structured((x_range, y_range))

    return field, sigma


def kriging(api: API, dataset: DataUpload, expander = st.sidebar) -> None:
    """Enable a kriging interface"""
    # check if kriging is enabled
    if st.session_state.get('kriging_enabled', False):
        ok = expander.button('Disable kriging')
        if ok:
            st.session_state.kriging_enabled = False
            st.experimental_rerun()
    else:
        st.sidebar.write('You can run a basic kriging interpolation')
        run = expander.button('Enable')
        if run:
            st.session_state.kriging_enabled = True
            st.experimental_rerun()
        else:
            return
    
    # get kriging parameters
    krig_method = expander.selectbox('Kriging method', options=[m for m in KRIGING_METHODS.keys() if m != 'external'], format_func=lambda k: KRIGING_METHODS.get(k))
    krig_kw = {}
    if krig_method == 'simple':
        krig_kw['mean'] = expander.number_input('Mean of the field', value=0.0)
    elif krig_method == 'universal':
        FUNC = {'linear': 'Linear drift', 'quadratic': 'Quadratic drift'}
        krig_kw['drift_functions'] = expander.selectbox('Drift function', options=list(FUNC.keys()), format_func=lambda k: FUNC.get(k))
    else:
        krig_kw['unbiased'] = True
    
    # apply the kriging
    field, sigma = _apply_kriging(dataset, vario_params={k:v for k,v in st.session_state.items()}, grid_resolution=100, **krig_kw)

    # add the plot
    l, c, r = st.columns(3)

    # left
    l.markdown('### Kriging interpolation')
    fig = go.Figure(go.Heatmap(z=field, colorscale='Earth_r'))
    l.plotly_chart(fig, use_container_width=True)

    # center
    c.markdown('### Kriging error variance')
    fig = go.Figure(go.Heatmap(z=sigma))
    c.plotly_chart(fig, use_container_width=True)

    # right
    r.markdown('### Python sample code')
    c = _code_sample(**st.session_state)
    c += f"\n# create a gstools.Krige instance\nkrige = vario.to_gs_krige()\n\n#Create a grid\n"
    c += "x = np.linspace(vario.coordinates[:,0].min(), vario.coordinates[:,0].max(), 100)\n"
    c += "y = np.linspace(vario.coordinates[:,1].min(), vario.coordinates[:,1].max(), 100)\n"
    c += "\n#Apply\nfield, sigma = krige.structured((x, y))\n"
    r.code(c, language='python')
    

def base_variogram(dataset: DataUpload, **kwargs) -> skg.Variogram:
    # get the needed parameters
    bin_func = kwargs.get('bin_method', st.session_state.get('bin_method'))
    n_lags = kwargs.get('n_lags', st.session_state.get('n_lags', 10))
    maxlag = kwargs.get('maxlag', st.session_state.get('maxlag'))
    estimator = kwargs.get('estimator', st.session_state.get('estimator'))
    model = kwargs.get('model', st.session_state.get('model'))
    fit_method = kwargs.get('fit_method', st.session_state.get('fit_method'))
    fit_sigma = kwargs.get('fit_sigma', st.session_state.get('fit_sigma'))
    use_nugget = kwargs.get('use_nugget', st.session_state.get('use_nugget', False))
    _range = kwargs.get('range', st.session_state.get('range'))
    _sill = kwargs.get('sill', st.session_state.get('sill'))
    _nugget = kwargs.get('nugget', st.session_state.get('nugget'))
    _shape = kwargs.get('shape', st.session_state.get('shape'))

    # build the variogram
    vario = dataset.base_variogram(bin_func=bin_func, n_lags=n_lags, maxlag=maxlag)

    # check if the estimator exists
    if estimator is not None:
        vario.estimator = estimator
    
    # check if a model was set
    if model is not None:
        # apply and calc
        vario.model = model
        vario.use_nugget = use_nugget

        # check for manual fitting
        if fit_method == 'manual':
            p = {'method': 'manual', 'range': _range, 'sill': _sill}
            if use_nugget: p['nugget'] = _nugget
            if model in ('matern', 'stable'): p['shape'] = _shape
            vario.fit(**p)
        else:
            vario.fit_method = fit_method
            vario.fit_sigma = fit_sigma
    
    return vario


def variogram_plot(api: API, dataset: DataUpload, **kwargs) -> go.Figure:
    """"""
    # # get the needed parameters
    bin_func = st.session_state.get('bin_method', kwargs.get('bin_method'))
    estimator = st.session_state.get('estimator', kwargs.get('estimator'))
    model = st.session_state.get('model', kwargs.get('model'))

    # check if a variogram was passed
    if 'variogram' in kwargs:
        vario = kwargs['variogram']
    else:
        vario = base_variogram(dataset, **kwargs)

    # build a Figure    
    fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.0,
                specs=[[{}], [{'rowspan': 4}], [None], [None], [None]]
            )

    # check if binning was run
    if bin_func is None:
        fig.add_text(x=0.5, y=0.5, text='No binning method selected')
        return fig

    # binnig
    x_bins = vario.bins
    count = vario.bin_count

    # add the histogram
    fig.add_trace(go.Bar(x=x_bins, y=count, marker=dict(color='red'), name='Histogram'), row=1, col=1)

    # check if estimator exists
    if estimator is None:
        fig.add_trace(
            go.Scattergl(x=vario.distance, y=vario._diff, mode='markers', marker=dict(color='blue', opacity=0.3, size=3)),
            row=2, col=1
        )
        for b in vario.bins:
            fig.add_vline(x=b, line=dict(color='red', dash='dash', width=2), row=2, col=1)
    else:
        # add the empirical variogram
        fig.add_trace(
            go.Scatter(x=vario.bins, y=vario.experimental, mode='markers', marker=dict(color='blue', size=7), name='Empirical variogram'),
            row=2, col=1
        )

    # next check for model
    if model is not None:
        x = np.linspace(0, vario.bins[-1], 100)
        y = vario.fitted_model(x)

        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', line=dict(color='green'), name=f'{model.capitalize()} model'),
            row=2, col=1
        )

    # figure layout
    fig.update_layout(legend_orientation='h')

    # general label
    fig.update_xaxes(title_text='Lag [-]', row=2, col=1)
    fig.update_yaxes(title_text='Count pairs', row=1, col=1)
    if estimator is not None:
        fig.update_yaxes(title_text=f'Semivariance [{estimator}]', row=2, col=1)

    return fig


def main_app(api: API, **kwargs):
    st.set_page_config(page_title='Learn Variograms', layout='wide')
    # get url params
    url_params = st.experimental_get_query_params()
    for k, v in url_params:
        st.session_state[k] = v

    # first check for story mode
    check_story_mode(api)

    # add the logo
    st.sidebar.image("https://firebasestorage.googleapis.com/v0/b/hydrocode-website.appspot.com/o/public%2Fhydrocode_brand.png?alt=media")
    
    # get a dataset
    dataset = data_select(api)

    # create a container for the variogram parameters
    vario_expander = st.sidebar.expander('Variogram parameters', expanded=True)

    # Binning parameters
    binning(api, dataset, vario_expander)

    # Estimator
    estimator(api, dataset, vario_expander)

    # Fitting method
    fit_method(api, dataset, vario_expander)

    # Fitting
    fit_expander = st.sidebar.expander('Fitting parameters', expanded=True)
    fitting(api, dataset, fit_expander)
    
    # show the variogram
    variogram(api, dataset)

    # run some kriging
    kriging(api, dataset)

    # add a debugging view
    if kwargs.get('debug', url_params.get('debug', False)):
        e = st.expander('DEBUG')
        e.json(st.session_state)


if __name__=='__main__':
    api = API(db_name='data.db')
    main_app(api)
