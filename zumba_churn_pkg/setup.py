# setup.py

import os
from setuptools import setup, find_packages
from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent
# Directorio para los archivos de requerimientos
REQ_DIR = ROOT_DIR / 'requirements'

def get_version():
    with open(ROOT_DIR / "VERSION") as version_file:
        return version_file.read().strip()

def get_requirements(fname="requirements.txt"):
    with open(REQ_DIR / fname) as f:
        return [line for line in f.read().splitlines() if not line.startswith('#') and line.strip()]

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            relative_path = Path(path).relative_to(ROOT_DIR / 'zumba_churn_pkg')
            paths.append(os.path.join(str(relative_path), filename))
    return paths

extra_files = package_files(ROOT_DIR / 'zumba_churn_pkg' / 'assets')
extra_files.append('../config.yml')
extra_files.append('../VERSION')

setup(
    name='zumba_churn_pkg',
    version=get_version(),
    description='Modelo XGBoost para predicción de churn',
    author='Daniela Uscategui, Andres Alfonso, Francisco Amorocho',
    packages=find_packages(exclude=('tests',)),
    install_requires=get_requirements("requirements.txt"),
    include_package_data=True,
    package_data={'zumba_churn_pkg': extra_files},
    python_requires='>=3.8',
)
