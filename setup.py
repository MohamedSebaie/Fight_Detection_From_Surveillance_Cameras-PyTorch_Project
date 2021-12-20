from setuptools import setup, find_packages

setup(name='Trial1',
version='0.1',
author='Human Action Recognition team',
packages=find_packages(),
install_requires=[
    "torch",
    "torchvision",
    "numpy",
    "albumentations",
    "ipython",
    "moviepy",
    "opencv",
    "pandas",
    "protobuf",
    "pytube",
    "pytube3",

],
zip_safe=False,
)