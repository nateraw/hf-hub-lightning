from setuptools import find_packages, setup

setup(
    name='hf-hub-lightning',
    packages=find_packages(exclude=['examples']),
    version='0.0.2',
    license='MIT',
    description='Callback for pushing to Hugging Face Hub from PyTorch Lightning',
    author='Nathan Raw',
    author_email='naterawdata@gmail.com',
    url='https://github.com/nateraw/hf-hub-lightning',
    install_requires=['pytorch-lightning', 'huggingface-hub'],
)
