import os

'''
This sets up some of the directories that DANSy will look into for specific data. This will mostly be for the complete proteome analysis.
'''

# Global variables for the complete proteome on what default values will look like.
# Note for memory efficiency we save the adjacency as a json file as it is a sparse matrix.
DANSY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For the data files these need to be first set up prior to the experiment so that dansy knows where to look for them. This can be updated with the create_DANSy_dirs function
directory_loc = f"{DANSY_DIR}/dansy/directories.txt"
dirs = []
with open(directory_loc,'r') as f:
    for line in f:
        dirs.append(line)
        
DANSY_DATA_DIR = dirs[0]

# This tells dansy what the current reference file of the proteome version from CoDIAC ends with (e.g. files that end with YYYYMMDD.csv) within the DANSY_DATA folder. This can be updated using the update_proteome_version function.
current_version_file = f'{DANSY_DIR}/dansy/current_proteome_version.txt'
prot_ver = []
with open(current_version_file, 'r') as f:
    for line in f:    
        prot_ver.append(line)
DANSY_PROTEOME_VERSION = prot_ver[0]

def create_DANSy_dirs(target_dir):
    '''
    Checks and/or creates a DANSY_DATA folder in the provided directory that DANSy will look at for any general data.

    Parameters
    ----------
     target_dir: str
        The directory where the DANSy data folder should be generated in and/or checked for the existence of
    '''

    install_dir = f"{target_dir}/DANSY_DATA/"
    if not os.path.exists(install_dir):
        # Make the directory
        os.makedirs(install_dir, exist_ok=True)

    if os.path.exists(install_dir):
        # Now update the directories file
        with open(f"{DANSY_DIR}/dansy/directories.txt",'w') as f:
            f.write(install_dir)

def update_proteome_version(version, default = True):
    '''
    Updates the proteome reference file version that will be used for analysis (i.e. imported for DANSy to use).

    Parameters
    ----------
    version: str
        The suffix (or full name) of the file(s) for the default reference files that dansy will look at.
    default: bool (optional)
        Whether the change in version is to be kept as the new default version used afterwards
    '''
    global DANSY_PROTEOME_VERSION
    DANSY_PROTEOME_VERSION = version
    if default:
        with open(f"{DANSY_DIR}/dansy/current_proteome_version.txt", 'w') as f:
            f.write(version)
        print(f'Updated the default proteome file to look for {version}')
