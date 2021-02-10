from distutils.core import setup
from setuptools import setup, find_packages


# with open('requirements.txt') as requirements_file:
#     install_requirements = requirements_file.read().splitlines()


setup(
  name         = 'fin_app_model',
  description  = 'fin_app_model',
  url          = 'https://github.com/pondelion/FinAppModel',
  packages     = ['fin_app_models'],
  # install_requires = install_requirements,
)
