from setuptools import setup

setup(
    name='heliostack',
    version='0.0.1',
    description='A Python library for prototyping long-baseline shift-and-stack.',
    author='Kevin J. Napier',
    author_email='kjnapier@umich.edu',
    url="https://github.com/kjnapier/heliostack",
    packages=['heliostack'],
    install_requires=['torch >= 2.0.0',
                      'numpy']
)