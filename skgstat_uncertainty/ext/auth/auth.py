import os
from cryptography.fernet import Fernet
import hashlib
from datetime import timedelta as td

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, User

DEFAULTPATH = os.path.abspath(os.path.join(os.path.expanduser('~'), 'skgstat_users.db'))

# load the encryption key and add to User class
from .encrypt_key import ENCRYPT_KEY
User.ENCRYPT_KEY = ENCRYPT_KEY

def get_session():
    # get uri
    uri = os.getenv('DB_URI', 'sqlite:///' + DEFAULTPATH)
    engine = create_engine(uri)

    # if sqlite db and it does not exist, create it
    if uri.startswith('sqlite'):
        if not os.path.exists(DEFAULTPATH):        
            Base.metadata.create_all(bind=engine)
    
    Session = sessionmaker(bind=engine)
    return Session()


def login(username: str = None, password: str = None, token: str = None, valid_for=td(days=14)) -> dict:
    session = get_session()

    # check if password or token login
    if username is not None and password is not None:
        user: User = session.query(User).filter(User.username == username).one()
        if not user.authenticated(password):
            raise RuntimeError('Invalid password')
    elif token is not None:
        phash = hashlib.sha256(token.encode()).hexdigest()
        user: User = session.query(User).filter(token==phash).one()
    else:
        raise ValueError('Either username and password or token must be provided')
    
    # create a session token
    session_token = user.session_login(valid_for=valid_for)

    return session_token


def verify_token(token: str) -> dict:
    session = get_session()
    session_token = User.verify_token(session, token)
    return session_token


def refresh_token(token: str) -> dict:
    session = get_session()
    session_token = User.refresh_token(session, token)
    return session_token


def register(username: str = None, password: str = None, token: str = None, alias: str = None, **kwargs) -> dict:
    # create a database session
    session = get_session()

    userdata = {}
    
    # check auth data
    if username is not None and password is not None:
        userdata['username'] = username
        userdata['password'] = hashlib.sha256(password.encode()).hexdigest()
    elif token is not None:
        userdata['token'] = hashlib.sha256(token.encode()).hexdigest()
    else:
        raise ValueError('Either username and password or token must be provided')
    
    # check extra data
    if alias is not None:
        userdata['alias'] = alias
    if 'db_name' in kwargs:
        userdata['db_name'] = kwargs['db_name']
    else:
        userdata['db_name'] = f'u_{username.capitalize() if username is not None else token.upper()}.db'
    if 'can_upload' in kwargs:
        userdata['can_upload'] = kwargs['can_upload']
    if 'share' in kwargs:
        userdata['share'] = kwargs['share']

    # create the user
    user = User(**userdata)
    try:
        session.add(user)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e

    # build the session login
    session_token = user.session_login()
    return session_token()
