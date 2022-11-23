from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_dict = generate_distutils_setup(
    packages=['control_force_provider'],
    package_dir={'': 'src'}
)

setup(**setup_dict)
