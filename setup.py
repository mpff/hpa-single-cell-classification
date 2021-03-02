from setuptools import find_packages, setup

setup(
    name='singl',
    packages=find_packages(),
    version='0.1',
    description='Scripts and modules for the HPA Single Cell Classification challenge on Kaggle.',
    author='Manuel Pfeuffer',
    license='BSD-3-Clause',
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        compress_dataset=singl.scripts.compress_dataset:main
    '''
)