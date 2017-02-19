from setuptools import setup, find_packages

readme = '''
mymllib
'''

license = '''
license
'''

setup(
    name='mymllib',
    version='0.0.1',
    description='mymllib',
    long_description=readme,
    author='roronya',
    author_email='roronya628@gmail.com',
    url='https://github.com/roronya/mymllib',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy', 'sklearn'
    ]
)
