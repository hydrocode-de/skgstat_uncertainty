import os
import pickle
import dash
import json
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import skgstat as skg
import numpy as np
import plotly.graph_objects as go

# path 
PATH = os.path.abspath(os.path.dirname(__file__))

# build the core application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

# gobally build the Variogram
# get the data
coordinates, values = skg.data.pancake(N=150, seed=42).get('sample')

# estimate the variogram
vario = skg.Variogram(coordinates, values, n_lags=20)
vario.maxlag = 500


# Controls
CONTROLS = dbc.Container(
    children=[
        dbc.Alert(
            "Variogram model parameters successfully saved!",
            id='alert',
            dismissable=True,
            fade=True,
            is_open=False,
            color='success'
        ),
        dcc.Dropdown(
            id='data-select',
            options=[
                {'label': 'Precise observations', 'value': '0'},
                {'label': '2/256 observation uncertainty', 'value': '2'},
                {'label': '5/256 observation uncertainty', 'value': '5'},
                {'label': '10/256 observation uncertainty', 'value': '10'},
                {'label': '15/256 observation uncertainty', 'value': '15'},
                {'label': '25/256 observation uncertainty', 'value': '25'}
            ],
            value='10'
        ),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Spherical model', 'value': 'spherical'},
                {'label': 'Exponential model', 'value': 'exponential'},
                {'label': 'Gaussian model', 'value': 'gaussian'},
                {'label': 'Matérn model', 'value': 'matern'},
                {'label': 'Cubic model', 'value': 'cubic'},
                {'label': 'Stable model', 'value': 'stable'}
            ],
            value='spherical'
        ),
        dbc.Row(
            children=[
                dbc.Col(html.H3('Effective Range'), sm=4),
                dbc.Col(html.H3('Model Sill and Nugget'), sm=4),
                dbc.Col(html.H3('Shape parameter'), sm=4)
            ],
            className='mt-3'
        ),
        dbc.Row([
            dbc.Col(
                children=[
                    dcc.RangeSlider(
                        id='r',
                        min=1,
                        max=500,
                        value=[100]
                    )
                ],
                sm=4
            ),
            dbc.Col(
                children=[
                    dcc.RangeSlider(
                        id='sill',
                        min=0.0,
                        max=2000.,
                        value=[0, 500.]
                    )
                ], 
                sm=4
            ),
            dbc.Col(
                children=[
                    dcc.RangeSlider(
                        id='shape',
                        min=0.2,
                        max=20,
                        value=[2],
                        step=0.1
                    )
                ], 
                sm=4
            )
        ]),
        dbc.Jumbotron([
            html.H3(
                children=[
                    html.Span('Parameters'),
                    html.Span('bb', id='r2')
                ],
                style=dict(justifyContent='space-between', display='flex')
            ),
            html.Pre(html.Code(id='text-output')),
            dbc.Button('Save', id='save', color='success')
        ])
    ],
    fluid=False,
    className='pt-5'
)

# main layout
LAYOUT = dbc.Container(
    children=[
        dcc.Store(id='sigma-store', storage_type='session'),
        dcc.Loading(dcc.Graph(id='main-figure')),
        CONTROLS
    ],
    fluid=True
)

# add the Layout
app.layout = LAYOUT


# helper functions
def load_data(sigma_value):
    with open(os.path.join(PATH, '..', 'data', f'MC_results_50000_{sigma_value}.pickle'), 'rb') as f:
        _d =  pickle.load(f)
        ebins = _d['ebins']
        eparams = _d['eparams']
        emodels = _d['emodels']
    
    return ebins, eparams, emodels

# Callbacks
@app.callback(
    Output('sigma-store', 'data'),
    Input('data-select', 'value')
)
def set_sigma_value(sigma_value):
    return sigma_value


@app.callback(
    Output('shape', 'disabled'),
    Input('model-selector', 'value')
)
def disable_shape(model_name):
    return model_name not in ('matern', 'stable')


@app.callback(
    Output('main-figure', 'figure'),
    Output('text-output', 'children'),
    Output('r2', 'children'),
    Input('sigma-store', 'data'),
    Input('model-selector', 'value'),
    Input('r', 'value'),
    Input('sill', 'value'),
    Input('shape', 'value')

)
def main_plot(sigma_value, model_name, r, sill_nugget, shape):
    # first load the data
    ebins, _, _ = load_data(sigma_value)

    # calculate absoulte error margins
    error_bounds = np.column_stack((
        np.min(ebins, axis=1),
        np.max(ebins, axis=1)
    ))

    # load the model
    model = getattr(skg.models, model_name)

    # create the x-values
    x = np.linspace(0, vario.maxlag, 100)

    # build the args array
    args = [r[0], sill_nugget[1]]
    if model_name in ('matern', 'stable'):
        args.append(shape[0])
    args.append(sill_nugget[0])
    
    # apply the model
    y = model(x, *args)

    # get the goodness of fit
    ystar = model(vario.bins, *args)
    gof = [1 if _y >= bnd[0] and _y <= bnd[1] else 0 for _y, bnd in zip(ystar, error_bounds)]
    se = [0 if _y >= bnd[0] and _y <= bnd[1] else np.min((np.abs(_y - bnd[0]), np.abs(_y - bnd[1])))**2 for _y, bnd in zip(ystar, error_bounds)]
    rmse = np.sqrt(np.mean(se))

    # build text output
    text = json.dumps({
        'model': model_name,
        'sigma_obs': int(sigma_value),
        'effective_range': r[0],
        'sill': sill_nugget[1],
        'nugget': sill_nugget[0],
        'shape': shape[0] if model_name in ('matern', 'stable') else 'n.a.',
        'fit': np.sum(gof) / len(vario.bins) * 100,
        'rmse': rmse
    }, indent=4)

    # build the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,0], mode='lines', line=dict(color='grey'), fill='none', name='lower bound')
    )
    fig.add_trace(
        go.Scatter(x=vario.bins, y=error_bounds[:,1], mode='lines', line=dict(color='grey'), fill='tonexty', name='upper bound')
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', line=dict(color='green', width=3), name=f'{model_name.capitalize()} model')
    )

    fig.update_layout(
        template='plotly_white',
        legend=dict(
            orientation='h'
        )
    )

    return fig, text, 'fit: %.1f%% [RMSE: %.1f]' % (np.sum(gof) / len(vario.bins) * 100, rmse)


@app.callback(
    Output('alert', 'is_open'),
    Input('save', 'n_clicks'),
    State('model-selector', 'value'),
    State('text-output', 'children'),
)
def on_save(n, model_name, text):
    fname = f'{model_name}.json'
    path = os.path.join(PATH, '..', 'data/model_fits', fname)

    # only run 
    if n is not None and n > 0:
        # load model file if it already exists
        if os.path.exists(path):
            with open(path, 'r') as f:
                store = json.load(f)
        else:
            store = []
        
        obj = json.loads(text)
        store.append(obj)

        # save
        with open(path, 'w') as f:
            json.dump(store, f, indent=4)
    
        return True
    else:
        return False


# main entrypoint
def run_server(host='127.0.0.1', port=8050, debug=False):
    """
    Run the dash application
    """
    app.run_server(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import fire
    fire.Fire(run_server)
