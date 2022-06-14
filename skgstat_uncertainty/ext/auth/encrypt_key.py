import os
from cryptography.fernet import Fernet


ENCRYPT_KEY = os.getenv('ENCRYPT_KEY')
if ENCRYPT_KEY is None:
    if os.path.exists(os.path.join(os.path.dirname(__file__), '.ENCRYPT_KEY')):
        with open(os.path.join(os.path.dirname(__file__), '.ENCRYPT_KEY')) as f:
            ENCRYPT_KEY = f.read()
    else:
        ENCRYPT_KEY = Fernet.generate_key().decode()
        with open(os.path.join(os.path.dirname(__file__), '.ENCRYPT_KEY'), 'w') as f:
            f.write(ENCRYPT_KEY) 
ENCRYPT_KEY = ENCRYPT_KEY.encode()