#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      defra
#
# Created:     10/10/2024
# Copyright:   (c) defra 2024
# Licence:     MIT
#-------------------------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name='my_package',  # Nom de ton package
    version='1.0.0',    # Version du package
    packages=find_packages(),  # Trouve tous les sous-packages
    install_requires=[
        'torch',        # Dépendances de ton projet
        'opencv-python',
        'torchvision',
        'numpy',
        'shapely',
        'pillow',
    ],
    entry_points={
        'console_scripts': [
            'run_my_package = scripts.my_script:main',  # Lien vers ton script principal
        ],
    },
    #description='A package for detecting letters Z, Q, S, D using PyTorch and OpenCV',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    url='https://github.com/tonnomdeutilisateur/my_package',  # URL vers le dépôt de ton projet (si applicable)
    author='Ton Nom',
    author_email='tonemail@example.com',
    license='MIT',  # Licence de ton projet
)