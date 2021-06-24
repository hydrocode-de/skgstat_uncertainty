import fire
import os
import subprocess


def spawn_app(name: str = 'index.py'):
    path = os.path.dirname(__file__)
    fname = os.path.join(path, 'apps', name)
    process = subprocess.Popen(['streamlit', 'run', fname])
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print('User requested exit')


fire.Fire(spawn_app)