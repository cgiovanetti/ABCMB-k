from setuptools import setup

setup(
    name='ABCMB',
    version='0.1.1.1',    
    description='A fast, differentiable, and extensible CMB code',
    url='https://github.com/TonyZhou729/ABCMB',
    author='Zilu Zhou, Cara Giovanetti, Hongwan Liu',
    author_email='cgiovanetti@lbl.gov',
    license='MIT',
    packages=['abcmb'],
    install_requires=['numpy',
                    'scipy',
                    'matplotlib',
                    'diffrax',
                    'equinox',
                    'interpax',
                    'jax==0.8.1',
                    'pytest'            
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14'
    ],
)