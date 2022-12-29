from distutils.core import setup

setup(
    name='ccm_memory',
    version='0.0.1',
    packages=['ccm_memory'],
    url='github',
    license='MIT',
    author='Carleton Cognitive Modeling Lab',
    author_email='renanozen@cmail.carleton.ca',
    description='',
    install_requires=[
        'numpy>=1.21.6',
        'hrrlib @ https://github.com/ren-oz/hrrlib/archive/refs/heads/main.zip#egg=hrrlib-0.0.1',
    ],
)
