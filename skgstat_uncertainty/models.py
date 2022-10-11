from typing import List, Tuple, Union
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship, object_session
from skgstat import Variogram
import numpy as np
import pyproj

from skgstat_uncertainty.processor import utils

# database Base model
Base = declarative_base()

# target coordinate reference system
TGT_CRS = pyproj.CRS.from_epsg(4326)


class CoverageData(Base):
    """"""
    __tablename__ = 'coverages'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False, unique=True)
    coverage_type = sa.Column(sa.Enum('Coverage', 'CoverageCollection'), nullable=False)
    sample_type = sa.Column(sa.Enum('sample', 'field', 'auxiliary', 'generic', 'interpolation', 'simulation'), nullable=False)
    domain = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=True)
    parameters = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False, default={})
    ranges = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)
    extra = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False, default={})

    @classmethod
    def from_sample(cls, x: np.ndarray, y: np.ndarray, *v: List[np.ndarray], parameters: dict = None, **kwargs) -> 'CoverageData':
        """"""
        # create base object
        coverageData = CoverageData(coverage_type='CoverageCollection', sample_type='sample')

        # set parameters
        coverageData.set_parameters(parameters=parameters)

        # build the ranges, will become coverages
        coverages = []
        for x_, y_, vals in zip(x, y, zip(*v)):
            coverages.append({
                'type': 'Coverage',
                'domain': {
                    'type': 'Domain',
                    'axes': {
                        'x': {'values': [x_]},
                        'y': {'values': [y_]}
                    }
                },
                'ranges': {par_name: {
                    'type': 'NdArray',
                    'dataType': 'integer' if isinstance(val, int) else 'float',
                    'values': [ val ]
                } for par_name, val in zip(self.parameters.keys(), vals)}
            })
        
        # for samples, the coverages are stored as ranges, as only the range and domain axes differ
        coverageData.ranges = coverages

        # set extra information
        coverageData.extra = kwargs

        return coverageData

    @classmethod
    def from_grid(cls, z: np.ndarray, xx: Unp.ndarray, yy: np.ndarray, parameters: dict = None, **kwargs) -> 'CoverageData':
        """"""
        # create base object
        coverageData = CoverageData(coverage_type='CoverageCollection', sample_type='sample')

        # set parameters
        coverageData.set_parameters(parameters=parameters)

        # set domain
        coverageData.domain = {
            'type': 'Domain',
            'domainType': 'Grid',
            'axes': {
                'x': {'start': xx.min(), 'stop': xx.max(), 'num': xx.size},
                'y': {'start': yy.min(), 'stop': yy.max(), 'num': yy.size}
            }
        }

        # iterate z
        if not isinstance(z, list):
            z = [z]

        # set range
        coverageData.ranges = {
            par_name: {
                'type': 'NdArray',
                'dataType': 'float',
                'axisNames': ['x', 'y'],
                'shape': [var.shape[0], var.shape[1]],
                'values': var.flatten().tolist()
            } for par_name, var in zip(coverageData.parameters.keys(), z)
        }

        # add extra information
        coverageData.extra = kwargs

        return coverageData
        
    def set_parameters(self, parameters: Union[dict, int, List[str]] = None, update: bool = False):
        """"""
        # build the parameters
        if parameters is None:
            parameters = {
                "Variable": {
                    'type': 'Parameter',
                    'description': {'en': 'Generic parameter of any observable. No further information is available.'},
                    'observedProperty': {'label': 'Unkown Property'}
                }
            }
        elif isinstance(parameters, int):
            num_of_pars = parameters
            parameters = {}
            for i, par in enumerate(num_of_pars):
                parameters[f"Variable{i + 1 if i > 0 else ''}"] = {
                    'type': 'Parameter',
                    'description': {'en': 'Generic parameter of any observable. No further information is available.'},
                    'observedProperty': {'label': 'Unkown Property'}
                }
        elif isinstance(parameters, list) and all([isinstance(_, str) for _ in parameters]):
            names = parameters
            parameters = {}
            for name in names:
                parameters[name] = {
                    'type': 'Parameter',
                    'description': {'en': f'Generic parameter of any observable called "{name}".'},
                    'observedProperty': {'label': name.capitalize()}
                }
        
        # update or save
        if update:
            old_params = self.parameters
        else:
            old_params = {}

        old_params.update(parameters)
        self.parameters = old_params

    def save(self):
        """Save current state of the coverage info"""
        # get a session
        session = object_session(self)
        
        # save changes
        try:
            session.add(self)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
    
    @property
    def data(self):
        """"""
        data = self.extra

        # get the main parameter and its data
        d = self.get_parameter_data(self.parameters.keys()[0])
        
        # add         
        if self.coverage_type == 'Coverage':
            data['field'] = d
        elif self.coverage_type == 'CoverageCollection':
            data['x'] = d[0]
            data['y'] = d[1]
            data['v'] = d[2]
        
        # return
        return data

    def get_parameter_data(self, parameter: str) -> Union[np.ndarray, Tuple[list, list, list]]:
        """"""
        # check if the parameter exists
        if parameter not in self.parameters:
            raise KeyError(f"{parameter} not in CoverageData.parameters")

        # check coverage type:
        cov_type = self.domain.get('type')

        # switch
        if cov_type == 'Coverage':
            r = self.ranges(parameter)
            data = np.asarray(r['values']).reshape(r['shape'])
        elif cov_type == 'CoverageCollection':
            # container
            x, y, v = list, list, list
            
            # extract the sample
            for cov in self.ranges:
                ax = cov['domain']['axes']
                x.append(ax['x']['values'][0])
                y.append(ax['y']['values'][0])
                v.append(cov['ranges'][parameter]['values'][0])
            
            # merge back
            data = (x, y, v, )

        return data

    def to_json(self) -> dict:
        """Return CoverageJSON of this coverage data"""
        cov = {
            'type': self.coverage_type,
            'parameters': self.parameters,
            'extra': self.extra
        }

        if self.coverage_type == 'Coverage':
            cov['domain'] = self.domain
            cov['ranges'] = self.ranges
        elif self.coverage_type == 'CoverageCollection':
            cov['domainType'] = 'Point'
            cov['coverages'] = self.ranges
        
        return cov

    def base_variogram(self, parameter: str = None, **kwargs) -> Variogram:
        # get the base parameter
        par = parameter if parameter is not None else self.parameters.keys()[0]
        
        if self.coverage_type == 'CoverageCollection':
            coords, vals = self.get_parameter_data(parameter=par)
        elif self.coverage_type == 'Coverage':
            # get the field
            field = self.get_parameter_data(parameter=par)
            
            # get the coordinates from domain
            x_ax = self.domain['axes']['x']
            if 'start' in self.domain['axes'][]
            x_ = np.linspace(self.domain['axes']['x']['start'], self.domain['axes']['x']['end'], self.domain['axes']['x']['num'])

class DataUpload(Base):
    """
    Base model for representing datasets in SkGstat-Uncertainty.
    A data upload into the database is represented by an id, name,
    data type and an arbitrary json-seriallizable objects to respresent
    the data itself and the associated metadata.
    The two fundamental data types are 'field' for areal information,
    organized as raster and 'sample' for collections of sample points.

    Data Attributes
    ---------------
    x, y, v : List, optional
        List of integers or floats representing 2D sample points ('x', 'y')
        and the associated observations ('v')
    field : List[List], optional
        List of List of integeger or floats representing a row-oriented
        2D field
    field_id : int
        If data type is 'sample' and a field in the database was sub-sampled
        to derive this dataset, the parent entitiy can be referenced by its id
    origin : str
       Origin information about the dataset. Markdown is allowed.
    description : str
        Description of the databse. Markdown is allowed.
    license : str
        License abbreviation for this dataset. Supported are:
        
        * no - No license - ask for permission
        * cc0 - Creative Commons public dedication
        * ccby - Creative Commons by Attribution
        * dldeby - Data license Germany - attribution
        * dlde0 - Data license Germany - zero
    doi : str
        DOI of a dataset description publication associated to this instance.
    crs : int
        EPSG number of a coordinate reference system associated to the sample
        or field. As of now, only samples are supported.

    """
    __tablename__ = 'uploads'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    upload_name = sa.Column(sa.String, nullable=False, unique=True)
    data_type = sa.Column(sa.String, nullable=False)
    data = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    variograms = relationship("VarioParams", back_populates="data", cascade="all, delete")

    def to_dict(self):
        """Return the instance data as dicionary"""
        return {
            'id': self.id,
            'upload_name': self.upload_name,
            'data_type': self.data_type,
            'data': self.data
        }

    def base_variogram(self, **kwargs) -> Variogram:
        """Return a baseic skg.Variogram from the given kwargs."""
        # coordinates and values
        coords = list(zip(*[self.data[dim] for dim in ('x', 'y', 'z') if dim in self.data]))
        values = self.data['v']

        # bug fix for n_lags
        if 'n_lags' in kwargs and kwargs['n_lags'] is None:
            del kwargs['n_lags']

        # instantiate the Variogram
        vario = Variogram(coordinates=coords, values=values, **kwargs)

        return vario

    def update_thumbnail(self, height: int = 200, width: int = 200, **kwargs):
        """Adds a preview of the data as thumbnail to the the specs"""
        # create the thumbnail
        thumbnail = utils.create_thumbnail(self, return_type='base64', height=height, width=width, **kwargs)

        # extract my data:
        data_spec = self.data
        
        # add the thumbnail to the data_spec
        data_spec['thumbnail'] = thumbnail

        # overwrite data specs
        self.data = data_spec

        # get a session and save
        session = object_session(self)
        try:
            session.add(self)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e

    def to_wgs84(self) -> Tuple[list, list]:
        """
        Convert the dataset to WGS84 referenced dataset, to use them on a web map.
        For this to work properly, as of now, the DataUpload needs a data_type of
        'sample' and a 'CRS' identifier has to be added to :attr:`DataUpload.data`
        """
        if self.data_type == 'sample' and 'crs' in self.data:
            # get source crs
            src_crs = pyproj.CRS.from_epsg(self.data['crs'])

            # create a transformer
            transformer = pyproj.Transformer.from_crs(src_crs, TGT_CRS, always_xy=True)

            x, y = transformer.transform(self.data['x'], self.data['y'])

            return x, y
        else:
            print("Only supported for DataUpload.data_type=='sample' with defined CRS.")


class VarioParams(Base):
    __tablename__ = 'variograms'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    data_id = sa.Column(sa.Integer, sa.ForeignKey('uploads.id'), nullable=False)
    name = sa.Column(sa.String, nullable=False)
    description = sa.Column(sa.String, nullable=True)
    params = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    data = relationship("DataUpload", back_populates="variograms")
    conf_intervals = relationship("VarioConfInterval", back_populates="variogram", cascade="all, delete")

    def to_dict(self):
        return {
            'id': self.id,
            'data_id': self.data_id,
            'name': self.name,
            'description': self.description,
            'params': self.params
        }
    
    @property
    def variogram(self) -> Variogram:
        # load the data from database
        data = self.data.data

        # get coords and vals
        coords = list(zip(*[data.get(dim) for dim in ('x', 'y', 'z') if dim in data]))
        values = data['v']

        # instantiate the variogram
        vario = Variogram(coords, values, **self.params)

        # return 
        return vario


class VarioConfInterval(Base):
    __tablename__ = 'variogram_conf_intervals'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    vario_id = sa.Column(sa.Integer, sa.ForeignKey('variograms.id'), nullable=False)
    name = sa.Column(sa.String, nullable=False)
    spec = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    variogram = relationship("VarioParams", back_populates="conf_intervals")
    models = relationship("VarioModel", back_populates='confidence_interval', cascade='all, delete')

    def to_dict(self):
        return {
            'id': self.id,
            'vario_id': self.vario_id,
            'name': self.name,
            'spec': self.spec
        }
    
    @property
    def interval(self):
        return self.spec['interval']
    
    def get_result_type(self, type_name: str) -> List['VarioModelResult']:
        result_list = []

        for model in self.models:
            result_list.extend([result for result in model.results if result.content_type == type_name])

        return result_list
        
    @property
    def kriging_fields(self):
        return self.get_result_type('kriging_field')
    
    @property
    def simulation_fields(self):
        return self.get_result_type('simulation_field')


class VarioModel(Base):
    __tablename__ = 'variogram_models'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    conf_id = sa.Column(sa.Integer, sa.ForeignKey('variogram_conf_intervals.id'), nullable=False)
    model_type = sa.Column(sa.String, nullable=False)
    parameters = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    confidence_interval = relationship("VarioConfInterval", back_populates='models')
    results = relationship("VarioModelResult", back_populates='model', cascade='all, delete')

    @property
    def variogram(self) -> Variogram:
        # create the skgstat.Variogram instance
        vario = self.confidence_interval.variogram.variogram
        
        # set model
        vario.model = self.model_type
        
        # extract the parameters
        params = self.parameters['model_params']
        vario.fit(
            method='manual',
            range=params.get('range'),
            sill=params.get('sill'),
            nugget=params.get('nugget'),
            shape=params.get('shape')
        )

        return vario

    @property
    def model_func(self):
        return self.variogram.model

    def apply_model(self, x: list) -> list:
        model = self.variogram.fitted_model

        return [model(_) for _ in x]

    def to_dict(self):
        return {
            'id': self.id,
            'conf_id': self.conf_id,
            'model_type': self.model_type,
            'parameters': self.parameters

        }

    def get_base_grid(self) -> Tuple[int, int]:
        """
        """
        # get the sample
        dataset = self.confidence_interval.variogram.data

        # check if a parent field exists
        if dataset.data.get('field_id', False):
            session = object_session(self)
            parent_dataset = session.query(DataUpload).filter(DataUpload.id == dataset.data['field_id']).one()
            parent_field = parent_dataset.data['field']
            return np.array(parent_field).shape
        
        # there is no parent filed
        else:
            x = dataset.data.get['x']
            y = dataset.data.get['y']

            return (np.max(x), np.max(y))

    def save(self):
        # get a session to the database
        session = object_session(self)

        try:
            session.add(self)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e


class VarioModelResult(Base):
    __tablename__ = 'variogram_model_results'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    model_id = sa.Column(sa.Integer, sa.ForeignKey('variogram_models.id'), nullable=False)
    content_type = sa.Column(sa.String, nullable=False)
    name = sa.Column(sa.String, nullable=True)
    content = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    model = relationship("VarioModel", back_populates="results")

    property
    def variogram(self) -> Variogram:
        return self.model.variogram
    
    def to_dict(self):
        out = {
            'id': self.id,
            'model_id': self.model_id,
            'content_type': self.content_type,
            'content': self.content
        }

        if self.name is not None:
            out['name'] = self.name
        
        return out