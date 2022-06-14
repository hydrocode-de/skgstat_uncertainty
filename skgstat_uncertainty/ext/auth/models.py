import hashlib
from cryptography.fernet import Fernet

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, object_session, Session
from datetime import datetime as dt
from datetime import timedelta as td


Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    ENCRYPT_KEY: bytes = None

    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String, unique=True, nullable=True)
    password = sa.Column(sa.String(64), nullable=True)
    token = sa.Column(sa.String(64), unique=True, nullable=True)
    alias = sa.Column(sa.String, nullable=True)
    db_name = sa.Column(sa.String, nullable=False)
    auth_enitity = sa.Column(sa.String, default='skgstat_ext')
    can_upload = sa.Column(sa.Boolean, nullable=False, default=True)
    share = sa.Column(sa.Boolean, nullable=False, default=False)

    def to_dict(self) -> dict:
        return dict(
            username=self.username,
            db_name=self.db_name,
            auth_entitiy=self.auth_enitity,
            can_upload=self.can_upload,
            share=self.share
        )
    
    def save(self):
        session = object_session(self)

        try:
            session.add(self)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
    
    def authenticated(self, password: str) -> bool:
        """Authenticate the entity"""
        # get the password hash
        phash = hashlib.sha256(password.encode()).hexdigest()

        # check if this is a password or token enitity
        if hasattr(self, 'password'):
            return self.password == phash
        else:
            return self.token == phash
    
    def session_login(self, valid_for=td(days=14)) -> dict:
        """Create a session token"""
        # valid
        valid_until = dt.utcnow() + valid_for
        # user-id::validity
        message = f'{self.id}::{int(valid_until.timestamp())}'
        session_token = Fernet(self.ENCRYPT_KEY).encrypt(message.encode()).decode() 

        # build the response
        return {
            'valid': valid_until,
            'token': session_token,
            'userdata': self.to_dict()
        }
    
    @classmethod
    def verify_token(cls, session: Session, token: str) -> dict:
        # decrpyt the token
        message = Fernet(cls.ENCRYPT_KEY).decrypt(token.encode()).decode()
        user_id, valid_until = message.split('::')
        valid_until = dt.utcfromtimestamp(int(valid_until))

        # check if valid
        if dt.utcnow() > valid_until:
            return {'error': True, 'message': 'Session token expired.'}
        
        # load the user
        try:
            user = session.query(User).filter(User.id==user_id).one()
        except Exception as e:
            return {'error': True, 'message': 'User not found.'}
        
        # return
        return {'error': False, 'valid': valid_until, 'userdata': user.to_dict()}
    
    @classmethod
    def refresh_token(cls, session: Session, token: str) -> dict:
        # decrypt the token
        message = Fernet(cls.ENCRYPT_KEY).decrypt(token.encode()).decode()
        user_id, _ = message.split('::')

        # load the user
        try:
            user = session.query(User).filter(User.id==user_id).one()
        except Exception as e:
            return {'error': True, 'message': 'User not found.'}
        
        # return
        self.session_login()
