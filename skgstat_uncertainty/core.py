from typing import Optional, Union, Literal, List
import os
import io
import glob
import json
import hashlib
import tarfile
from typing import List
import skgstat as skg
import numpy as np
import pandas as pd
from datetime import datetime as dt

from skgstat_uncertainty import templates
#from . import templates

class Project:    
    def __init__(
        self, 
        path: Optional[str] = os.path.join(os.path.dirname(__file__), 'projects', 'Collection of Uncertain Pancakes'),
        variogram: Optional[str] = 'dc2d24c0a4603c57cf729f156ffa97c6',
        sigma: Union[int, float] = 5,
        std_level: float = 1.5,
        std_target: Literal['min', 'max', 'mean'] = 'min',
        model_evalf: Literal['rmse', 'cv'] = 'rmse',
        filter_include_fit: bool = True,
        exclude_model_ids: List[int] = [],
        n_iterations: int = 50000,
    ):
        """
        Project management
        """
        self.path = path
        self._name = None

        # currently cached values
        self._vario = variogram
        self._sigma = sigma
        self.n_iterations = n_iterations
        self._std_level = std_level
        self._filter_include_fit = filter_include_fit
        self._filter_include_level = 98
        self._exclude_model_ids = exclude_model_ids

        # set the filter for model params
        self._filtered_models = None
        self._cached_model_fields = None

        # set these after the fields and model caches are created
        self.std_target = std_target
        self.model_evalf = model_evalf

        # monte-carlo container
        self._mc_output_data = None

        # try to load a name
        if self.name is None and path is not None:
            self.name = f'Uncertainty_Analysis_%s' % (dt.utcnow().strftime('%Y_%m_%d_%H_%M_%S'))

    @property
    def data_path(self):
        if self.path is None:
            raise RuntimeError('Project is not initialized.')
        return os.path.join(self.path, 'data')

    @property
    def result_path(self):
        if self.path is None:
            raise RuntimeError('Project is not initialized')
        return os.path.join(self.path, 'results')

    @property
    def result_base_name(self):
        fname = f"{self._vario}_{self.n_iterations}_{self.sigma}_%s"
        return os.path.join(self.result_path, fname)

    @property
    def fpath_config(self):
        return os.path.join(self.data_path, 'config.json')

    @property
    def fpath_variograms(self):
        return os.path.join(self.data_path, 'variograms.json')

    @property
    def fpath_variogram_data(self):
        return os.path.join(self.data_path, f'{self._vario}_variogram_data.npz')
    
    @property
    def fpath_model_fits(self):
        return os.path.join(self.data_path, 'model_fits.json')

    @property
    def initialized(self):
        # without path this project is not valid
        if self.path is None or self.name is None:
            return False

        # check for files
        fnames = ['config.json', 'variograms.json', 'model_fits.json']
        return all([os.path.exists(os.path.join(self.data_path, fname)) for fname in fnames])
    
    @property
    def name(self):
        # if set return
        if self._name is not None:
            return self._name

        # if no path and no name, return None
        if self.path is None:
            return None

        # load missing name if given
        name = self.config().get('project_name')
        if name is not None:
            self.name = name
        
        return self._name

    @name.setter
    def name(self, new_name: str):
        # set the new name
        self._name = new_name

        # save to config file
        conf = self.config()
        conf['project_name'] = self._name
        self.config(new_conf=conf)

    @property
    def vario(self) -> skg.Variogram:
        return self.get_variogram(self._vario)
    
    @vario.setter
    def vario(self, new_md5):
        self._vario = new_md5

        # reset cache
        self._filtered_models = None
        self._cached_model_fields = None
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, new_sigma: int):
        self._sigma = new_sigma

        # reset cache
        self._filtered_models = None
        self._cached_model_fields = None

    @property
    def std_level(self):
        return self._std_level
    
    @std_level.setter
    def std_level(self, new_std):
        self._std_level = new_std

        # reset cache
        self._filtered_models = None
        self._cached_model_fields = None

    @property
    def std_target(self):
        return self._std_target

    @std_target.setter
    def std_target(self, new_target):
        if new_target.lower() not in ('min', 'max', 'mean'):
            raise ValueError("new target has to be one of: ('min', 'max', 'mean')")
        
        self._std_target = new_target.lower()

        # reset filter
        self._filtered_models = None
        self._cached_model_fields = None

    @property
    def model_evalf(self):
        return self._model_evalf

    @model_evalf.setter
    def model_evalf(self, new_evalf):
        if new_evalf.lower() not in ('rmse', 'cv'):
            raise ValueError("new evaluation function has to be one of ('rmse', 'cv')")

        self._model_evalf = new_evalf.lower()

        # reset filter
        self._filtered_models = None
        self._cached_model_fields = None

    @property
    def filter_include_fit(self):
        return self._filter_include_fit
    
    @filter_include_fit.setter
    def filter_include_fit(self, new_state):
        self._filter_include_fit = new_state

        # reset cache
        self._filtered_models = None
        self._cached_model_fields = None

    @property
    def prefiltered_models(self):
        if self._filtered_models is None:
            self.prefilter_models()
        return self._filtered_models

    @property
    def excluded_model_ids(self):
        return self._exclude_model_ids
    
    @excluded_model_ids.setter
    def excluded_model_ids(self, new_list):
        self._exclude_model_ids = new_list

        # prefilter the kriged filed list
        self.load_kriging_cache(force_reload=True)

    @property
    def kriged_model_fields(self) -> dict:
        if self._cached_model_fields is None:
            self.load_kriging_cache()
        return self._cached_model_fields

    @property
    def kriged_field_stack(self) -> np.ndarray:
        return np.stack(list(self.kriged_model_fields.values()), axis=2)

    @property
    def cached_fields(self) -> list:
        return list(self.kriged_model_fields.keys())
    
    @property
    def uncached_fields(self) -> list:
        cached = self.cached_fields
        return [p['md5'] for p in self.prefiltered_models if p['md5'] not in cached]

    def prefilter_models(self):
        """
        Filter model parameters and fill filtered_model array.
        All saved models are filtered for being less than 
        Project.std_level away from population mean or best RMSE.

        """
        # get all models
        all_models = self.load_model_params(sigma=self.sigma)

        # get the evalf to use
        evalf = self.model_evalf

        if len(all_models) > 0:
            # calculate stats
            rmse_std = np.std([p[evalf] for p in all_models])
            if self.std_target == 'max':
                criterion = np.max([p[evalf] for p in all_models])
            elif self.std_target == 'min':
                criterion = np.min([p[evalf] for p in all_models])
            else:
                criterion = np.mean([p[evalf] for p in all_models])

            self._filtered_models = [p for p in all_models if np.abs(criterion - p[evalf]) <= rmse_std * self.std_level or (self.filter_include_fit and p['fit'] >= self._filter_include_level)]
        else:
            self._filtered_models = []

    def load_kriging_cache(self, force_reload=False, check_excuded=True):
        # create cache 
        if self._cached_model_fields is None or force_reload:
            self._cached_model_fields = dict()
        
        # go for all prefiltered models
        for param in self.prefiltered_models:
            md5 = param['md5']
            
            # if exists, skipped
            if md5 in self._cached_model_fields:
                continue

            # if model params are excluded, skip
            if param['id'] in self.excluded_model_ids and check_excuded:
                continue

            # load
            try:
                field = self.load_single_kriging_field(md5)
                if field is None:
                    raise KeyError
                else:
                    self._cached_model_fields[md5] = field
            except KeyError:
                continue

    @classmethod
    def create(cls, name:str, path='./', add_sample=False, **kwargs) -> 'Project':
        """
        Create a new Project instance
        """
        # base config
        conf = {'sigma_levels': {}, 'project_name': name, 'variograms': []}

        # build the args
        args = {'path': os.path.join(path, name)}
        
        # create a base variogram if needed
        if add_sample:
            vario = Project.create_base_variogram(**kwargs)
            v_dict = Project.hash_variogram(vario)
            
            # add original field
            pan = skg.data.pancake_field(band=0).get('sample')
            v_dict['original_field'] = pan.tolist()

            _hash = list(v_dict.keys())[0]
            conf.update({
                'variograms': [
                    {
                        'md5': _hash, 
                        'name': 'Pancake', 
                        'description': f"Pancake variogram from SciKit-GStat using {kwargs.get('N', 150)} sample points at pseudo-random positions seeded with {kwargs.get('seed', 42)}"
                    }
                ]
            })
            
            # add args
            args['variogram'] = _hash
        else:
            v_dict = dict()

        # create the main project folder structure
        if not os.path.exists(os.path.join(path, name, 'data')):
            os.makedirs(os.path.join(path, name, 'data'))
        if not os.path.exists(os.path.join(path, name, 'results')):
            os.makedirs(os.path.join(path, name, 'results'))

        # config file
        with open(os.path.join(path, name, 'data', 'config.json'), 'w') as f:
            json.dump(conf, f, indent=4)
        
        # model fits file
        with open(os.path.join(path, name, 'data', 'model_fits.json'), 'w') as f:
            fits = {_hash: []} if add_sample else {}
            json.dump(fits, f, indent=4)

        # store the file
        with open(os.path.join(path, name, 'data', 'variograms.json'), 'w') as f:
            json.dump(v_dict, f, indent=4)
        
        # create some static file contents
        with open(os.path.join(path, name, 'README.md'), 'w') as f:
            f.write(templates.README.format(project_name=name))
        with open(os.path.join(path, name, 'bibliography.bib'), 'w') as f:
            f.write(templates.BIBTEX)
        with open(os.path.join(path, name, 'LICENSE'), 'w') as f:
            f.write(templates.LICENSE)

        # create the instance and return
        return Project(**args)

    def save_base_variogram(
        self,
        vario: skg.Variogram,
        original: np.ndarray = None,
        name: str = None,
        description: str = None
    ):        
        # read files
        with open(self.fpath_variograms, 'r') as f:
            variograms = json.load(f)

        # read config
        conf = self.config()

        # hash the variogram and add
        v_dict = self.hash_variogram(vario=vario)
        # get the hash
        md5 = list(v_dict.keys())[0]
        
        if original is not None:
            v_dict[md5]['original_field'] = original.tolist()
        
        # update
        variograms.update(v_dict)

        # add to config
        name = name if name is not None else f"{v_dict.get('model').capitalize()} model"
        description = description if description is not None else f"{str(vario)} - {dt.utcnow().isoformat()}"
        conf['variograms'].append({
            'md5': md5,
            'name': name,
            'description': description
        })
        
        # save everything again
        self.config(new_conf=conf)
        
        with open(self.fpath_variograms, 'w') as f:
            json.dump(variograms, f, indent=4)

        return md5

    @classmethod
    def create_base_variogram(cls, N=150, seed=42, **kwargs) -> skg.Variogram:
        # get sample
        c, v = skg.data.pancake(N=N, seed=seed).get('sample')

        # estimate
        vario = skg.Variogram(c, v, **{k: v for k, v in kwargs.items() if k != 'maxlag'})
        vario.maxlag = kwargs.get('maxlag')

        return vario
    
    @classmethod
    def create_base_variogram_from_field(cls, field: np.ndarray, N=150, seed=42, **kwargs) -> skg.Variogram:
        # create the random generator
        rng = np.random.default_rng(seed)

        # create random index pairs without replace
        idx = rng.choice(np.multiply(*field.shape), replace=False, size=N)

        # build a mechgrid of coordinates
        _x, _y = np.meshgrid(*[range(dim) for dim in field.shape])
        x = _x.flatten()
        y = _y.flatten()

        # build variogram
        coords = np.asarray([[x[i], y[i]] for i in idx])
        vals = np.asarray([field[c[0], c[1]] for c in coords])
        vario = skg.Variogram(coords, vals, **{k: v for k, v in kwargs.items() if k != 'maxlag'})
        vario.maxlag = kwargs.get('maxlag')

        return vario

    @classmethod
    def hash_variogram(cls, vario: skg.Variogram) -> dict:
        # get the params
        p = vario.describe().get('params')

        # add the sample
        p['coordinates'] = vario.coordinates.tolist()
        p['values'] = vario.values.tolist()

        # hash
        md5 = hashlib.md5(json.dumps(p).encode()).hexdigest()

        return {md5: p}

    def config(self, new_conf: dict = None) -> dict:
        """
        """
        with open(self.fpath_config, 'r') as f:
            conf = json.load(f)

        # return if needed
        if new_conf is None:
            return conf
        
        # otherwise update
        conf.update(new_conf)

        with open(self.fpath_config, 'w') as f:
            json.dump(conf, f, indent=4)
        
        return conf

    def  get_variogram(self, md5: str) -> skg.Variogram:
        if os.path.exists(self.fpath_variograms):
            with open(self.fpath_variograms, 'r') as f:
                varios = json.load(f)
        else:
            varios = dict()
        
        # reconstruct the variogram
        params = varios.get(md5)
        
        # there is still a bug in metric space
        vario = skg.Variogram(**{k:v for k, v in params.items() if k not in ('maxlag', 'original_field')})
        vario.maxlag = params.get('maxlag')
        
        return vario
    
    def list_variograms(self) -> dict:
        with open(self.fpath_variograms, 'r') as f:
            varios = json.load(f)
        
        return varios
    
    @property
    def has_original_field(self):
        return self.original_field is None

    @property
    def original_field(self):
        try:
            with open(self.fpath_variograms, 'r') as f:
                varios = json.load(f)
                return np.asarray(varios[self._vario]['original_field'])
        except Exception:
            return None
    
    def truncated_original_field(self):
        # get the variogram
        vario = self.vario
        field = self.original_field

        # return None if there is no original
        if field is None:
            return None

        # span the grid for kriging        
        x = np.arange(np.min(vario.coordinates[:,0]), np.max(vario.coordinates[:,0]), 1)
        y = np.arange(np.min(vario.coordinates[:,1]), np.max(vario.coordinates[:,1]), 1)

        # select from the grid
        xx, yy = np.meshgrid(x, y)
        return field[xx,yy].T.copy()

    def monte_carlo_simulation(self, N: int = 50000, sigma: int = 5, seed: int = 42) -> np.ndarray:
        # get the variogram
        vario = self.vario
        fixed_variogram_parameters = vario.describe().get('params')

        # create the input field and result container
        input_data = np.array([vario.values, ] * N)
        self._mc_output_data = np.ones((vario.n_lags, N)) * np.nan

        # choose new input data from a seeded PDF
        rng = np.random.default_rng(seed)
        input_data = input_data.astype(float) + (rng.random(size=input_data.shape) - 0.5) * (1 + sigma)

        # extract the MetricSpace from Variogram to speed things up
        ms = vario.metric_space

        # main loop
        for i, values in enumerate(input_data):
            # calculate the new variogram
            v = vario.clone()
            v.values = values
            # v = skg.Variogram(ms, values, **fixed_variogram_parameters)
            self._mc_output_data[:, i] = v.experimental

            yield i + 1

    def save_error_bounds(self, sigma: int = 5, name=None, description=None):
        # get the filename
        store_fname = os.path.join(self.data_path, f'{self._vario}_{self.n_iterations}_error_bounds.npz')

        # get simulation family
        N = self.n_iterations

        # get the result
        simulation_result = self._mc_output_data
        if simulation_result is None:
            raise RuntimeError('You first need to run a simulation')

        # Save Simulation
        # open
        if os.path.exists(store_fname):
            all_bounds = dict(np.load(store_fname, allow_pickle=True))
        else:
            all_bounds = dict()

        # add the simulation
        all_bounds[f'level_{sigma}'] = simulation_result

        # save again
        np.savez(store_fname, **all_bounds)

        # handle error bounds
        error_bounds = np.column_stack((
            np.min(simulation_result, axis=1),
            np.max(simulation_result, axis=1)
        ))

        # update name and description
        if name is None:
            name = f'{sigma}/256 observation uncertainty'
            description = f'Observation uncertainty of {sigma}/256 value space'

        # update the config
        conf = self.config()
        
        # handle saving
        it_name = f'{N}_iterations'
        if self._vario not in conf['sigma_levels'].keys():
            conf['sigma_levels'].update({self._vario: {it_name: []}})
        if it_name not in conf['sigma_levels'][self._vario].keys():
            conf['sigma_levels'][self._vario].update({it_name: []})

        levels = conf['sigma_levels'][self._vario][it_name]
        levels.append({
            'level': sigma,
            'name': name,
            'description': description,
            'bounds': error_bounds.tolist()
        })
        
        # append and save
        conf['sigma_levels'][self._vario][it_name] = levels
        self.config(new_conf=conf)

        # empty the cache
        self._mc_output_data = None

    def get_error_levels(self, as_dict=False):
        """
        """
        levels = self.config().get('sigma_levels', {}).get(self._vario, {}).get(f'{self.n_iterations}_iterations', [])

        if as_dict:
            return {l['level']: l['name'] for l in levels}
        else:
            return [l['level'] for l in levels]

    def load_error_bounds(self, sigma: int = None):
        levels = self.config().get('sigma_levels', {}).get(self._vario, {}).get(f'{self.n_iterations}_iterations', [])

        if sigma is None:
            return {l['level']: np.array(l['bounds']) for l in levels}
        else:
            bnd = [l['bounds'] for l in levels if l['level'] == sigma][0]
            return np.array(bnd)

    def monte_carlo_result_table(self) -> pd.DataFrame:
        all_bnds = self.load_error_bounds(sigma=None)
        info = list() 
        
        # get all simulations at given iterations
        for k, v in all_bnds.items():
            d = {
                'level': k,
                'mean conf. interval': np.mean([b[1] - b[0] for b in v]),
                'min conf. interval': np.min([b[1] - b[0] for b in v]),
                'max conf. interval': np.max([b[1] - b[0] for b in v])
            }
            info.append(d)
        
        # turn into dataframe
        result_table = pd.DataFrame(info)        
        return result_table

    def apply_variogram_model(self, params: dict):
        """
        Appply a theoretical variogram model.
        """
        vario = self.vario
        
        # get x
        x = np.linspace(0, vario.maxlag, 100)

        # load model
        model = getattr(skg.models, params.get('model'))

        # build args
        args = [params.get('effective_range'), params.get('sill')]
        if params.get('shape') != 'n.a.':
            args.append(params.get('shape'))
        args.append(params.get('nugget'))
        
        # apply model
        y = model(x, *args)

        return x, y

    def create_model_params(self, model_name: str, range: float, sill: float, nugget=0, shape=None, cross_validate=False) -> dict:
        # laod the variogram
        vario = self.vario.clone()
        
        # get x
        x = np.linspace(0, vario.maxlag, 100)

        # load model
        model = getattr(skg.models, model_name)

        # build args
        args = [range, sill]
        if shape is not None:
            args.append(shape)
        args.append(nugget)

        x = vario.bins
        y = model(x, *args)
        
        # get the error bounds
        bounds = self.load_error_bounds(self.sigma)

        # get fit
        fit = [1 if _y >= bnd[0] and _y <= bnd[1] else 0 for _y, bnd in zip(y, bounds)]
        se = [0 if _y >= bnd[0] and _y <= bnd[1] else np.min(((_y - bnd[0])**2, (_y - bnd[1])**2)) for _y, bnd in zip(y, bounds)]

        # build params
        params = {
            'model': model_name,
            'sigma_obs': self.sigma,
            'effective_range': range,
            'sill': sill,
            'nugget': nugget,
            'shape': shape if shape is not None else 'n.a.',
            'fit': float(np.sum(fit) / len(fit) * 100),
            'rmse': float(np.sqrt(np.mean(se)))
        }

        # make a cross validation, this can take some time
        if cross_validate:
            # update the variogram copy
            vario._kwargs.update(dict(fit_range=range, fit_sill=sill, fit_nugget=nugget, fit_shape=shape))
            vario.model = model_name
            vario.fit_method = 'manual'
            
            # perform cross-validation
            cv = vario.cross_validate(metric='mae')
            params['cv'] = cv

        # claculate the md5 hash for all
        md5 = hashlib.md5(json.dumps(params).encode()).hexdigest()
        params['md5'] = md5

        return params
    
    def save_model_params(self, params: dict, if_exists='raise'):
        with open(self.fpath_model_fits, 'r') as f:
            all_fits = json.load(f)
        
        # load existing
        fits = all_fits.get(self._vario, [])

        # check if the params already exist
        if any([p['md5'] == params['md5'] for p in fits]):
            if if_exists == 'raise':
                raise AttributeError('Parameters already exist')
        else:
            # append an ID and current N
            params['n_iterations'] = self.n_iterations
            params['id'] = len(fits) + 1

            # save
            fits.append(params)
            all_fits[self._vario] = fits
        
        # save
        with open(self.fpath_model_fits, 'w') as f:
            json.dump(all_fits, f, indent=4)

    def load_model_params(self, sigma: int = None, model_name: str = None):
        # load the file
        with open(self.fpath_model_fits, 'r') as f:
            fits = json.load(f)
        
        # get the variogram params
        all_params =  fits.get(self._vario, [])

        # always apply the filter for correct simulation family
        all_params = [p for p in all_params if p['n_iterations'] == self.n_iterations]

        # apply filters if given
        if sigma is not None:
            all_params = [p for p in all_params if p['sigma_obs'] == sigma]
        
        if model_name is not None:
            all_params = [p for p in all_params if p['model'] == model_name]
        
        # return 
        return all_params

    def get_params(self, params_md5: str) -> dict:
        all_models = self.load_model_params()

        try:
            params = [p for p in all_models if p['md5'] == params_md5][0]
        except KeyError:
            return None

        return params

    def model_params_count(self):
        return len(self.load_model_params(sigma=self.sigma))
    
    def filtered_model_params(self, model_name: str = None, std_level: float = None, include_fits = True):
        pass
    
    def apply_kriging(self, params: dict):
        # get a clone of the variogram to change it
        vario = self.vario.clone()

        # set the model
        vario.model = params.get('model')

        # build the argument array
        args = dict(
            fit_range=params.get('effective_range'),
            fit_sill=params.get('sill'),
            fit_nugget=params.get('nugget')
        )
        if params.get('model') in ('matern', 'stable'):
            args['fit_shape'] = params.get('shape')
        
        # update the kwargs directly
        vario._kwargs.update(args)

        # use manual fitting
        vario.fit_method = 'manual'

        # build a grid
        x = np.arange(np.min(vario.coordinates[:,0]), np.max(vario.coordinates[:,0]), 1)
        y = np.arange(np.min(vario.coordinates[:,1]), np.max(vario.coordinates[:,1]), 1)

        # apply kriging
        krige = vario.to_gs_krige()
        field, _ = krige.structured((x, y))

        return field
    
    def save_kriging_field(self, field: np.ndarray, param_md5: str):
        fname = os.path.join(self.data_path, f'{self._vario}_kriging_fields.npz')

        # load all saved files
        fields = self.load_all_kriging_fields()

        # add 
        fields[param_md5] = field

        # save back
        np.savez(fname, **fields)

    def load_all_kriging_fields(self, sigma: int = None, model_name: str = None) -> dict:
        fname = os.path.join(self.data_path, f'{self._vario}_kriging_fields.npz')
        
        # if the file does not exists, return an empty dict
        if not os.path.exists(fname):
            return {}
        
        # load the needed params
        if sigma is not None or model_name is not None:
            params = self.load_model_params(sigma=sigma, model_name=model_name)
            fields = {p.get('md5'): self.load_single_kriging_field(p.get('md5')) for p in params}
        else:
            fcontent = np.load(fname, allow_pickle=True)
            fields = dict(fcontent)
        
        return fields

    def load_single_kriging_field(self, params_md5: str) -> np.ndarray:
        fname = os.path.join(self.data_path, f'{self._vario}_kriging_fields.npz')
        
        # if the file does not exists, return an empty dict
        if not os.path.exists(fname):
            return None
        
        # open the container
        fcontent = np.load(fname)
        return fcontent[params_md5]

    def kriged_field_conf_interval(self, lower: int, higher: int) -> np.ndarray:
        # get the stack
        fs = self.kriged_field_stack

        # calculate percentiles over axis 2
        lower, higher = (
            np.percentile(fs, lower, axis=2),
            np.percentile(fs, higher, axis=2)
        )

        # only stack the fields within the conf interval
        stack = np.stack([fs[:,:,i] * (fs[:,:,i] >= lower).astype(int) * (fs[:,:,i] <= higher).astype(int) for i in range(fs.shape[2])], axis=2)
        mean = np.mean(stack, axis=2)
        std = np.std(stack, axis=2)
        count = np.count_nonzero(stack, axis=2)

        return lower, higher, mean, std, count
        
    def kriged_fields_info(self, lower: int, higher: int) -> List[dict]:
        # result container
        single_fields = []
        
        # load currently cached fields
        all_fields = self.kriged_model_fields

        # check if an original field is available
        original = self.truncated_original_field()

        for md5, field in all_fields.items():
            params = self.get_params(md5)

            # base information
            d = {
                'id': params.get('id'),
                'model': params.get('model').capitalize(),
                'model fit': '%.1f %%' % params.get('fit'),
                'model fit RMSE': '%.1f' % params.get('rmse'),
                'in interval': '%.1f %%' % (np.sum((field >= lower) & (field <= higher).astype(int)) / (np.multiply(*field.shape)) * 100),
                'value range': '[%d, %d]' % (int(np.min(field)), int(np.max(field)))
            }


            if original is not None:
                d['field RMSE'] = '%.1f' % (np.sqrt(np.sum(np.power(field - original, 2))))
                d['variance recover'] = '%.1f %%' % (np.var(field) / np.var(original) * 100)

            single_fields.append(d)
        
        return single_fields

    def save(self, path_or_buffer: Union[str, io.BytesIO] = None):
        """
        """ 
        # create a filename:
        if path_or_buffer is None:
            path_or_buffer = os.path.join(self.data_path, '..', 'projects', self.name)

        if isinstance(path_or_buffer, str):
            if not path_or_buffer.endswith('.tar.gz'):
                path_or_buffer += '.tar.gz'
            
        # get the content of path
        fnames = glob.glob(os.path.join(self.data_path, '*'))
        fnames.extend(glob.glob(os.path.join(self.data_path, 'data', '*')))
        fnames.extend(glob.glob(os.path.join(self.data_path, 'results', '*')))

        # create the tarball
        if isinstance(path_or_buffer, io.BytesIO):
            tar = tarfile.open(fileobj=path_or_buffer, mode='w:gz')
        else:
            tar = tarfile.open(path_or_buffer, 'w:gz')

        # add the files
        for fname in fnames:
            # only npz, json, bib, and result files and md are saved
            if fname.split('.')[-1] in ('npz', 'json'):
                tar.add(fname, arcname='data/' + os.path.basename(fname))
            elif fname.split('.')[-1] in ('bib', 'md'):
                tar.add(fname, arcname=os.path.basename(fname))
            elif fname.split('.')[-1].lower() in ('png', 'txt', 'tex', 'pdf', 'svg'):
                tar.add(fname, arcname='results/' + os.path.basename(fname))
        
        # add  .gitkeep to the eventually empty results folder
        tar.addfile(tarfile.TarInfo('/results/.gitkeep'), io.BytesIO(b''))

        if isinstance(path_or_buffer, str):
            tar.close()
            return path_or_buffer
        else:
            tar.close()
            path_or_buffer.seek(0)
            return path_or_buffer
    
    @classmethod
    def open(cls, path_or_buffer: Union[str, io.BytesIO], extract_path: str = None, project_name: str = None):
        if isinstance(path_or_buffer, str):
            if not path_or_buffer.endswith('.tar.gz'):
                path_or_buffer += '.tar.gz'
            
            if not os.path.exists(path_or_buffer):
                raise FileNotFoundError(f'The specified <Project {path_or_buffer}> does not exist')
            
            # read from file system
            tar = tarfile.open(path_or_buffer, mode='r:gz')
        else:
            path_or_buffer.seek(0)
            tar = tarfile.open(fileobj=path_or_buffer, mode='r:gz')

        # handle proejct_name
        if project_name is None:
            if isinstance(path_or_buffer, str):
                project_name = os.path.basename(path_or_buffer)
            else:
                path_or_buffer.seek(0)
                project_name = hashlib.md5(path_or_buffer.read()).hexdigest()
                path_or_buffer.seek(0)
        
        # manage extract path
        if extract_path is None:
            extract_path = os.path.join(os.path.dirname(__file__), 'projects', project_name)
        else:
            extract_path = os.path.join(extract_path, project_name)
        
        # create extracted project dir
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        
        # extract
        tar.extractall(extract_path)

        return Project(path=extract_path)
    
    @classmethod
    def list_extracted_projects(cls) -> dict:
        # get the extract path for projects
        extract_path = os.path.join(os.path.dirname(__file__), 'projects')

        # scan for projects
        dirs = [d for d in os.scandir(extract_path) if d.is_dir()]

        # create the output dir
        return {d.name: d.path for d in dirs}

    def current_results(self, level='variogram'):
        # everything for current variogram
        return glob.glob(self.result_path + f'/{self._vario}_*')

    def all_results(self):
        return glob.glob(self.result_path + '/*')

    def delete_results(self):
        pass
