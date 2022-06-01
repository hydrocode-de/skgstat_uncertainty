from typing import List

from skgstat_uncertainty.models import VarioModel
from skgstat_uncertainty.api import API

ADD_BTN = "data:image/png;base64,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"


def variomodel_to_dict(models: List[VarioModel], add_measures = False) -> List[dict]:
    # build up the data for the table
    data = list()

    for model in models:
        d = {'id': model.id}

        # get the parameters
        d.update(model.parameters.get('model_params', {}))

        if add_measures:
            d.update(model.parameters.get('measures', {}))
        # append
        data.append(d)

    return data


def card_options_from_dataset_names(_api: API, datasets: dict, add_button: bool = True) -> List[dict]:
    """
    Transform the dataset names into a list of options, which can be consumed by the
    streamlit_card_select component.

    """
    if add_button:
        # create a container with an add button
        options = [dict(option='new', title='Create new dataset', image=ADD_BTN)]
    else:
        options = []
    
    # add each dataset as preview
    for data_id, name in datasets.items():
        d = dict(option=data_id, title=f"[ID: {data_id}] {name}")

        # get the dataset from db
        dataset = _api.get_upload_data(id=data_id)

        # handle description
        d['description'] = dataset.data['description'][:250] if 'description' in dataset.data else '<i>This dataset has no description</i>'

        if 'thumbnail' in dataset.data:
            d['image'] = dataset.data['thumbnail']
        
        # insert
        options.insert(0, d)

    # return
    return options


# add some constants
FIT_METHODS = {
    'trf': 'Trust-Region Reflective',
    'lm': 'Levenberg-Marquardt',
    'ml': 'Parameter Maximum Likelihood',
    'manual': 'Manual Fitting' 
}

MODELS = {
    'spherical': 'Spherical',
    'exponential': 'Exponential',
    'gaussian': 'Gaussian',
    'cubic': 'Cubic',
    'stable': 'Stable',
    'matern': 'Matérn'
}

BIN_FUNC = dict(
    even='Evenly spaced bins',
    uniform='Uniformly distributed bin sizes',
    kmeans='K-Means clustered bins',
    ward='hierachical clustered bins',
    sturges="Sturge's rule binning",
    scott="Scott's rule binning",
    sqrt="Squareroot rule binning",
    fd="Freedman-Diaconis estimator binning",
    doane="Doane's rule binning"
)

ESTIMATORS = dict(
    matheron="Matheron estimator",
    cressie="Cressie-Hawkins estimator",
    dowd="Dowd estimator",
    genton="Genton estimator",
    entropy="Shannon entropy"
)

MAXLAG = dict(
    median="Median value",
    mean="Mean value",
    ratio="Ratio of max distance",
    absolute="Absolute value",
    none="Disable maxlag"
)

KRIGING_METHODS = dict(
    simple='Simple Kriging',
    ordinary='Ordinary Kriging',
    universal='Universal Kriging',
    external='External drift Kriging'
)

CONF_METHODS = dict(
    std="Sample standard deviation inference",
    kfold="Bootstraped k-fold cross-validation",
    absolute="Observation uncertainty propagation (MC)",
    residual="Residual extrema elimination",
)

LICENSES = dict(
    no="No License - Contact owner for permission",
    cc0="Creative Commons Public Domain Dedication",
    ccby="Creative Commons Attribution 4.0 International",
    dldeby="Data license Germany - attribution - version 2.0",
    dlde0="Data license Germany - Zero - version 2.0"
)

PERFORMANCE_MEASURES = dict(
    rmse='Parameter fit - RMSE',
    cv='Model - Cross-validation',
    dic='Model type - DIC',
    er='Parameter fit - Empirical Risk',
    srm='Combined - Structural Risk Minimization',
)
