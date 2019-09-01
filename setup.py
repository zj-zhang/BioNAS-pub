from setuptools import setup, find_packages

config = {
    'include_package_data': True,
    'description': 'Bioinformatics Neural Architecture Search',
    'download_url': 'https://github.com/zj-zhang/BioNAS-pub',
    'version': '0.0.1',
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'package_data': {'BioNAS':['resources/mock_black_box/tmp_*/*', 'resources/simdata/*']},
    'setup_requires': [],
    'install_requires': ['numpy', 'matplotlib', 'scipy', 'tensorflow>=1.9.0,<2.0.0', 'keras>=2.0.0', 'seaborn>=0.9.0'],
    'dependency_links': [],
    'scripts': ['bin/BioNAS'],
    'name': 'BioNAS',
}

if __name__== '__main__':
    setup(**config)
