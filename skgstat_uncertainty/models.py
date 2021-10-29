from typing import List, Tuple
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from skgstat import Variogram


Base = declarative_base()


class DataUpload(Base):
    __tablename__ = 'uploads'

    # columns
    id = sa.Column(sa.Integer, primary_key=True)
    upload_name = sa.Column(sa.String, nullable=False, unique=True)
    data_type = sa.Column(sa.String, nullable=False)
    data = sa.Column(MutableDict.as_mutable(sa.JSON), nullable=False)

    # relationships
    variograms = relationship("VarioParams", back_populates="data", cascade="all, delete")

    def to_dict(self):
        return {
            'id': self.id,
            'upload_name': self.upload_name,
            'data_type': self.data_type,
            'data': self.data
        }


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
    
    @property
    def kriging_fields(self):
        result_list = []

        for model in self.models:
            result_list.extend([result for result in model.results if result.content_type == 'kriging_field'])

        return result_list


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