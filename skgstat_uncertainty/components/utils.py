from typing import List

from skgstat_uncertainty.models import VarioModel


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
