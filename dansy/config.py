import os

'''
This sets up some of the directories that DANSy will look into for specific data. This will mostly be for the complete proteome analysis.
'''

# Global variables for the complete proteome on what default values will look like.
# Note for memory efficiency we save the adjacency as a json file as it is a sparse matrix.
DANSY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANSY_DATA_DIR = DANSY_DIR + '/DANSY_DATA'
DANSY_PROTEOME_VERSION = '20250512.csv'

def create_DANSy_dirs(target_dir=None):

    install_dir = target_dir if target_dir else DANSY_DATA_DIR
    os.makedirs(install_dir, exist_ok=True)

def update_proteome_version(version):
    global DANSY_PROTEOME_VERSION 
    DANSY_PROTEOME_VERSION = version


if __name__ == '__main__':
    create_DANSy_dirs()
