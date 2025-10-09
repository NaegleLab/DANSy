from setuptools import setup, find_packages

setup(
    name="dansy",
    version="0.1.0",
    author="Naegle Lab",
    author_email="kmn4mj@virginia.edu",
    url="https://github.com/NaegleLab/DANSy",
    install_requires = ['pandas==2.2.*', 'numpy==2.0.*', 'scipy==1.13.*','networkx==3.2.*', 'matplotlib==3.9.*','seaborn==0.13.*'],
    license="GNU General Public License v3",
    description="DANSy: Domain Architecture Network Syntax",
    long_description="DANSy is an open-source software to analyze protein domain architectures by combining n-gram analysis and network approaches. DANSy represent proteins as sentences made up of domain words and constructs networks on the domain combinations in a set of given proteins. DANSy can be used to analyze either groups of proteins or to analyze the separation and enrichment of domain n-grams from gene or protein expression datasets.",
    project_urls = {'Documentation':'https://naeglelab.github.io/DANSy/'},
    classifiers= [
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent"
        ],
    packages=find_packages(),
    include_package_data=True,
    python_requires = ">=3.9",
    zip_safe = False
    )