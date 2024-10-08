# setup.py

from setuptools import setup, find_packages

setup(
    name='SimpleCNN',
    version='0.1.0',
    description='An easy-to-use AI library for building neural networks.',
    author='Mustafa Bangash',
    author_email='mustafa22bangash@gmail.com',
    url='https://github.com/MustafaBangash/SimpleCNN',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='neural-networks deep-learning ai machine-learning',
    python_requires='>=3.6',
)
