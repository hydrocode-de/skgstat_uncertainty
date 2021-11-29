import streamlit as st
from sympy import Integer
import plotly.graph_objects as go

from skgstat_uncertainty.api import API
from skgstat_uncertainty import components, symbolic
from skgstat_uncertainty.models import VarioParams


def derive_model(vario: VarioParams):
    with st.spinner('Substituting data and solving derivate...'):
        # get the data
        lag_classes = []
        for lag in vario.variogram.lag_classes():
            if len(lag) < 1:
                lag_classes.append(None)
                continue
            # initialize with lag data
            estimator = _derive_estimator(lag)

            # append
            lag_classes.append(estimator)
    
    return lag_classes

@st.cache
def _derive_estimator(lags):
    est = symbolic.Matheron(data=lags)
    return est()

def estimate_uncertainty(lag_classes, delta_x):
    dx = Integer(delta_x) / Integer(1000)

    cols = st.columns(len(lag_classes))

    # for col, m in zip(cols, lag_classes):
    #     if m is None:
    #         col.write('NaN')
    #     else:
    #         col.write(f'{m} +/- {m * dx}')
    
    conf = [(float(m.evalf() - (m * dx).evalf()), float(m.evalf() + (m * dx).evalf())) if m is not None else [None, None] for m in lag_classes]
    return conf

def main_app(api: API):
    st.title('Symbolic Compare')
    st.warning('Currently, only Matheron estimated confidence intervals are supported.')

    # get the dataset to play with
    dataset, vario, conf_interval = components.data_selector(api=api, data_type='sample', container=st.sidebar)

    st.markdown("## Generic form\nFor all equations, $x := |Z(s_i) - Z(s_{i + h})|$\n")
    st.markdown("Current model:")
    
    # derive the correct estimator class here

    # TODO: this needs to be dynamic
    matheron = symbolic.Matheron()
    st.write(matheron.model)

    st.markdown('## Adjust error level')
    # set a error level
    l, r = st.columns((1, 6))
    l.empty()
    sig = r.slider('Error level', min_value=1, value=10, max_value=1000, step=1)
    l.markdown('### $\Delta x = \\frac{%d}{100}$' % (sig))

    lag_classes = derive_model(vario)
    conf = estimate_uncertainty(lag_classes, sig)

    st.markdown('## As plot')

    left, right = st.columns(2)

    conf_fig = components.base_conf_graph(vario=vario, interval=conf_interval)
    conf_fig.update_layout(title=f"Numerical method: {conf_interval.spec.get('method', 'unknown')}")
    left.plotly_chart(conf_fig, use_container_width=True)

    sym_fig = go.Figure()
    x = vario.variogram.bins
    sym_fig.add_trace(
        go.Scatter(x=x, y=[b[0] for b in conf], mode='lines', line_color='grey', fill=None, name=f'lower')
    )
    sym_fig.add_trace(
        go.Scatter(x=x, y=[b[1] for b in conf], mode='lines', line_color='grey', fill='tonexty', name=f'upper')
    )
    sym_fig.update_layout(
        title='Symbolic confidence interval',
        legend=dict(orientation='h'),
        xaxis=dict(title='Lag', showgrid=False),
        yaxis=dict(title=f"{vario.variogram.estimator.__name__.capitalize()} semi-variance", showgrid=False),
    )

    right.plotly_chart(sym_fig, use_container_width=True)


if __name__ == '__main__':
    api = API(data_path='/home/mirko/Dropbox/python/uncertain_geostatistics/data', db_name='u_Mirko.db')
    main_app(api=api)
