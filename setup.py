from setuptools import setup

setup(
    name='heliostack',
    version='0.0.1a',
    description='A Python Package for Solar System Ephemerides and Dynamics.',
    author='Kevin J. Napier',
    author_email='kjnapier@umich.edu',
    url="https://github.com/kjnapier/heliostack",
    install_requires=['torch',
                      'numpy']
)