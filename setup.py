from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().split('\n')


def version():
    with open('VERSION') as f:
        return f.read().strip()


def readme():
    with open('README.md') as f:
        return f.read().strip()


setup(
    name='skgstat_uncertainty',
    version=version(),
    license='MIT License',
    install_requires=requirements(),
    description='Uncertainty analysis tool for SciKit-GStat',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True
)
