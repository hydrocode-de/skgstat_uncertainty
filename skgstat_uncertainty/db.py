import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


from .models import Base

# TODO: put this into a config file 
DATAPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DBNAME = 'data.db'


def _get_sqlite_engine(**kwargs):
    # build the db uri
    data_path = kwargs.get('data_path', DATAPATH)
    db_name = kwargs.get('db_name', DBNAME)
    uri = f"sqlite:///{data_path}/{db_name}"

    # build the engine
    engine = create_engine(uri)

    # check if the file exists
    if not os.path.exists(os.path.join(data_path, db_name)):
        # check if the folder exists
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        Base.metadata.create_all(bind=engine)

    return engine

def get_session(uri: str = None, mode: str = 'session', **kwargs) -> Session:
    if uri is None:
        engine = _get_sqlite_engine(**kwargs)
    else:
        engine = create_engine(uri)
    
    # return 
    if mode.lower() == 'engine':
        return engine
    
    # create the session
    SessionCls = sessionmaker(bind=engine)
    return SessionCls()


def install(drop=False):
    # get an engine
    engine = get_session(mode='engine')

    # drop everyting if needed
    if drop:
        Base.metadata.drop_all(bind=engine)
    
    # create all
    Base.metadata.create_all(bind=engine)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'install': install
    })

