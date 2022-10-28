"""
Main data management API
------------------------
The datasets in the backend database of *SKGstat-Uncertainty* can be accessed and managed
using this API. On creation, you can pass the path or connection string to the backend database,
which defaults to the standard SQLite database distributed with the package.

.. note::
    Before you can connect a backend database other than the default SQLite options, like a 
    PostgreSQL or MySQL backend, you need to run the 
    :func:`install function <skgstat_uncertainty.db.install>` for this backend.

If there is anything the API can't do for you, the underlying database session pool and 
database object models can easily be accessed. Refer to SQLAlchemy for more details.

Example:

.. code-block:: python
    api = API()

    session = api.session  # sqlalchemy.orm.Session

    # now you can execute SQL queries
    with session.connect() as con:
        res = con.execute('SELECT count(*) FROM uploads;')

    # you can also access all models
    from skgstat_uncertainty.models import DataUpload

    datasets = session.query(DataUpload).limit(5).all()

"""
from typing import Dict, List, Tuple, Union
from functools import wraps

from .db import get_session
from .models import DataUpload, VarioParams, VarioConfInterval, VarioModel, VarioModelResult

class API:
    def __init__(self, **kwargs):
        """
        Create a API instance. The kwargs accept anything that `create_engine` from 
        SQLAlchemy accepts. Additionally, there are some extra config options:

        Parameters
        ----------
        uri : str, optional
            Connection URI. Works for almost any relational database system.
            If not supplied, the SQLite backend will be used.
        data_path : str, optional
            Optional path to the directory containing the SQLite databases.
            If uri is supplied, data_path will be ignored. In any other case
            it defaults to the data path of the repository.
        db_name : str, optional
            Optional name of the SQLite database file. If uri is supplied,
            db_name will be ignored. In any other case it defaults to ``'data.db'``

        """
        self._kwargs = kwargs

        if 'uri' in self._kwargs:
            self.session = get_session(self._kwargs['uri'], mode='session')
        else:
            # open a database session
            self.session = get_session(uri=None, mode='session', **{k: v for k, v in self._kwargs.items() if k in ('db_name', 'data_path')})

    def get_upload_names(self, data_type: Union[List[str], str] = ['sample', 'field']) -> Dict[int, str]:
        """
        Load all IDs and names for all datasets in the database.
        Can be used to let the user select a dataset from a dropdown, before loading
        all data into memory.

        Parameters
        ----------
        data_type : list
            List of DataUpload.data_types that should be considered. By omitting
            'auxiliary' and 'simulation_field' (default) you can skip these kind of
            intermediate datasets.
        
        Returns
        -------
        names : dict
            Dictionary of {id: name} for all filtered datasets.
        """
        # get base query
        query = self.session.query(DataUpload.id, DataUpload.upload_name)

        # check the type of data_type
        if isinstance(data_type, str):
            data_type = [data_type]
        
        # build the filter for data type
        if data_type is not None:
            query = query.filter(DataUpload.data_type.in_(data_type))
        
        return {row[0]: row[1] for row in query.all()}

    def filter_auxiliary_data(self, parent_id: int) -> List[DataUpload]:
        """
        Load auxiliary data from the dataset for a given parent DataUpload.id.

        .. note::
            To filter for the parent id, the API needs to extract JSON metadata
            for all 'auxiliary' instances in the database and thus load all
            dataset into memory. If there are many and large auxiliary datasets,
            this function might take some time.

        Parameters
        ----------
        parent_id : int
            The id of the parent dataset, the auxiliary dataset is belonging to.
        
        Returns
        -------
        datasets : list
            List of DataUpload instances.

        """
        # build the base query
        query = self.session.query(DataUpload)
        
        # apply the filter
        query = query.filter(DataUpload.data_type=='auxiliary')

        return [dataset for dataset in query.all() if dataset.data.get('parent_id', -1) == parent_id]
    
    def get_upload_data(self, id=None, name=None) -> DataUpload:
        """
        Load a DataUpload of given id or name from the database.

        Parameters
        ----------
        id : int, optional
            ID of the DataUpload to be loaded.
        name : str, optional
            title of the DataUpload to be loaded

        """
        if id is None and name is None:
            raise AttributeError('Either id or name has to be given.')
        
        query = self.session.query(DataUpload)
        if id is not None:
            query = query.filter(DataUpload.id == id)
        else:
            query = query.filter(DataUpload.upload_name == name)

        return query.first()
    
    def set_upload_data(self, name, data_type, **data) -> DataUpload:
        """
        Upload a new or edit an existing DataUpload in the database and 
        return the new instance after mutation. The data-model is very 
        flexible and will store anything passed as keyword argument into
        a JSON field. You have to use specific keywords and formats,
        otherwise the processors can't make any sense of the data.

        Parameters
        ----------
        name : str
            A unique title for the dataset
        data_type : str
            Has to be one of 'field', 'sample', 'auxiliary' or 'simulation_field'

        """
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
    
    def update_upload_data(self, id, name: str = None, data_type: str = None, **data) -> DataUpload:
        # get the dataset
        dataset = self.get_upload_data(id=id)

        # set new values:
        if name is not None:
            dataset.upload_name = name
        if data_type is not None:
            dataset.data_type = data_type

        # get old data
        old_data = dataset.data.copy()

        # update data
        old_data.update(data)

        dataset.data = old_data

        try:
            self.session.add(dataset)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
        
        return dataset

    def delete_upload_data(self, id=None, name=None) -> None:
        """
        Delete the DataUpload of given ID or name.

        Parameters
        ----------
        id : int, optional
            ID of the DataUpload to be loaded.
        name : str, optional
            title of the DataUpload to be loaded

        """
        # get the dataset
        dataset = self.get_upload_data(id=id, name=name)

        try:
            self.session.delete(dataset)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def filter_vario_params(self, data_id=None, name=None) -> List[VarioParams]:
        """
        Filter the empirical variogram estimations stored in the 
        databse.

        Parameters
        ----------
        data_id : int, optional
            Filter VarioParams for the parenting DataUpload.id
        name : str, optional
            Filter VarioParams for title.

        """
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
