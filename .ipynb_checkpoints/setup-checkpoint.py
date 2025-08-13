import setuptools 
import os


def read(fname):
    """Reads the contents of a file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# --- Project Metadata ---
VERSION = '0.1.0'
DESCRIPTION = 'DiscERN: Discoverer of Evolutionarily Related Natural products'

LONG_DESCRIPTION = read('README.md')
PACKAGE_NAME = 'DiscERN'
AUTHOR_NAME = 'Jeremy G. Owen'
AUTHOR_EMAIL = 'jeremy.owen@vuw.ac.nz'
PROJECT_URL = 'to add'

# --- Setup Configuration ---
setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown", # Specify README format
    url=PROJECT_URL,

    packages=setuptools.find_packages(),

    package_data={
        'discern': [
            'data/*.json', # Include all .json files in discern/data/
            'data/*.tsv',  # Include all .tsv files in discern/data/
            'data/*.pkl',  # Include all .pkl files in discern/data/
        ]
    },

    install_requires=["scikit-learn",
                      "matplotlib",
                      "scipy",
                      "numpy",
                      "pandas",
                      "matplotlib-venn",
                      "tqdm",
                      "biopython",
                      "antismash",
                      "pyhmmer==0.10.7"
                     ],

    entry_points={
        'console_scripts': [
            'discern=discern.discern:main', 
        ],
    },

)
