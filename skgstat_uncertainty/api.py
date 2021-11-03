from typing import List, Tuple, Union
from functools import wraps
from sqlalchemy import Integer

from .db import get_session
from .models import DataUpload, VarioParams, VarioConfInterval, VarioModel, VarioModelResult

class API:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

        if 'uri' in self._kwargs:
            self.session = get_session(self._kwargs['uri'], mode='session')
        else:
            # open a database session
            self.session = get_session(uri=None, mode='session', **{k: v for k, v in self._kwargs.items() if k in ('db_name', 'data_path')})

    def get_upload_names(self, data_type: Union[List[str], str] = None):
        # get base query
        query = self.session.query(DataUpload.id, DataUpload.upload_name)

        # always exculde auxiliary data
        query = query.filter(DataUpload.data_type.in_(['sample', 'field']))

        # apply filter
        if data_type is not None:
            if isinstance(data_type, str):
                query = query.filter(DataUpload.data_type==data_type)
            elif isinstance(data_type, list):
                query = query.filter(DataUpload.data_type.in_(data_type))
        
        return {row[0]: row[1] for row in query.all()}

    def filter_auxiliary_data(self, parent_id: int) -> List[DataUpload]:
        # build the base query
        query = self.session.query(DataUpload)
        
        # apply the filter
        query = query.filter(DataUpload.data_type=='auxiliary')

        return [dataset for dataset in query.all() if dataset.data.get('parent_id', -1) == parent_id]
    
    def get_upload_data(self, id=None, name=None) -> DataUpload:
        if id is None and name is None:
            raise AttributeError('Either id or name has to be given.')
        
        query = self.session.query(DataUpload)
        if id is not None:
            query = query.filter(DataUpload.id == id)
        else:
            query = query.filter(DataUpload.upload_name == name)

        return query.first()
    
    def set_upload_data(self, name, data_type, **data) -> DataUpload:
        # check if create or replace is needed
        dataset = self.session.query(DataUpload).filter(DataUpload.upload_name==name).first()

        if dataset is None:
            dataset = DataUpload()
        
        # set new values
        dataset.upload_name = name
        dataset.data_type = data_type
        dataset.data = data

        try:
            self.session.add(dataset)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
        
        return dataset

    def delete_upload_data(self, id=None, name=None) -> None:
        # get the dataset
        dataset = self.get_upload_data(id=id, name=name)

        try:
            self.session.delete(dataset)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def filter_vario_params(self, data_id=None, name=None) -> List[VarioParams]:
        # build the base query
        query = self.session.query(VarioParams)

        # apply filter
        if data_id is not None:
            query = query.filter(VarioParams.data_id == data_id)
        if name is not None:
            name = name.replace('*', '%')
            if '%' not in name:
                name = f'%{name}%'
            query = query.filter(VarioParams.name.like(name))
        
        # return
        return query.all()
        
    def get_vario_params(self, id=None, name=None) -> VarioParams:
        if id is None and name is None:
            raise AttributeError('Either id or name has to be given.')

        # build the query
        query = self.session.query(VarioParams)
        if id is not None:
            query = query.filter(VarioParams.id == id)
        else:
            query = query.filter(VarioParams.name == name)
        
        return query.first()

    def set_vario_params(self, name, params, data_id, description=None) -> VarioParams:
        # instatiate the model
        vario = VarioParams(name=name, params=params, data_id=data_id, description=description)

        try:
            self.session.add(vario)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

        return vario

    def delete_vario_params(self, id=None, name=None) -> None:
        # get the variogram
        vario = self.get_vario_params(id=id, name=name)

        try:
            self.session.delete(vario)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
    
    def get_conf_interval(self, id=None, name=None) -> VarioConfInterval:
        if id is None and name is None:
            raise AttributeError('Either id or name has to be given.')
        
        # build the query
        query = self.session.query(VarioConfInterval)
        if id is not None:
            query = query.filter(VarioConfInterval.id == id)
        else:
            query = query.filter(VarioConfInterval.name == name)
        
        return query.first()

    def set_conf_interval(self, name, vario_id, interval: List[Tuple[float, float]], **extra) -> VarioConfInterval:
        # build the spec object
        spec = dict(interval=[list(tup) for tup in interval])
        spec.update(extra)

        # instantiate the model
        conf_interval = VarioConfInterval(name=name, vario_id=vario_id, spec=spec)

        try:
            self.session.add(conf_interval)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

        return conf_interval

    def delete_conf_interval(self, id=None, name=None) -> None:
        # get the interval
        interval = self.get_conf_interval(id=id, name=name)

        try:
            self.session.delete(interval)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
    
    def filter_vario_model(self, conf_id=None, model_type=None) -> List[VarioModel]:
        # build the base query
        query = self.session.query(VarioModel)

        # add the filter
        if conf_id is not None:
            query = query.filter(VarioModel.conf_id == conf_id)
        if model_type is not None:
            query = query.filter(VarioModel.model_type == model_type)
        
        return query.all()

    def get_vario_model(self, id: int) -> VarioModel:
        # build the query
        query = self.session.query(VarioModel).filter(VarioModel.id == id)

        return query.one()

    def set_vario_model(self, conf_id: int, model_type: str, **params) -> VarioModel:
        # instantiate the model
        model = VarioModel(conf_id=conf_id, model_type=model_type, parameters=params)

        try:
            self.session.add(model)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

        return model
    
    def delete_vario_model(self, id) -> None:
        # get the model
        model = self.get_vario_model(id=id)

        try:
            self.session.delete(model)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
    
    def filter_results(self, model_id=None, conf_id=None, content_type=None) -> List[VarioModelResult]:
        # build the base query
        query = self.session.query(VarioModelResult)

        # add filter
        if model_id is not None:
            query = query.filter(VarioModelResult.model_id == model_id)
        if content_type is not None:
            query = query.filter(VarioModelResult.content_type == content_type)
        if conf_id is not None:
            query = query.join(VarioModel).filter(VarioModel.conf_id == conf_id)
        
        return query.all()

    def get_result_content_types(self) -> List[str]:
        query = self.session.query(VarioModelResult.content_type.distinct())
        return [row[0] for row in query.all()]

    def get_result(self, id=None, name=None) -> VarioModelResult:
        if id is None and name is None:
            raise AttributeError('Either id or name has to be given.')
        
        # build the query
        query = self.session.query(VarioModelResult)
        if id is not None:
            query = query.filter(VarioModelResult.id == id)
        else:
            query = query.filter(VarioModelResult.name == name)
        
        return query.first()

    def set_result(self, model_id: int, content_type: str, name=None, **content) -> VarioModelResult:
        # instantiate the model
        res = VarioModelResult(model_id=model_id, content_type=content_type, name=name, content=content)

        try:
            self.session.add(res)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
        
        return res
    
    def delete_result(self, id=None, name=None) -> None:
        # get the result
        result = self.get_result(id=id, name=name)

        try:
            self.session.delete(result)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e


def cli_formatter(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # catch the result
        result = f(*args, **kwargs)

        # format output
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        if isinstance(result, list):
            return [r.to_dict() if hasattr(r, 'to_dict') else r for r in result]
    return wrapper


class Cli:
    api = None
    
    def __getattribute__(self, method_name):
        # check if the API has that method
        if not hasattr(Cli.api, method_name):
            return lambda *args, **kwargs: f"{method_name} is not a valid API method"
        
        # get the method and wrap it with a formatter
        func = getattr(Cli.api, method_name)
        
        # decorate and return
        return cli_formatter(func)




if __name__ == '__main__':
    import fire
    Cli.api = API()
    fire.Fire(Cli)
