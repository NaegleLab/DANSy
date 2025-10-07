import os

'''
This sets up some of the directories that DANSy will look into for specific data. This will mostly be for the complete proteome analysis.
'''

# Global variables for the complete proteome on what default values will look like.
# Note for memory efficiency we save the adjacency as a json file as it is a sparse matrix.
DANSY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For the data files these need to be first set up prior to the experiment so that dansy knows where to look for them
directory_loc = f"{DANSY_DIR}/dansy/directories.txt"
dirs = []
with open(directory_loc,'r') as f:
    for line in f:
        dirs.append(line.split()[0])
        
DANSY_DATA_DIR = dirs[0]
DANSY_PROTEOME_VERSION = '20250512.csv'

def create_DANSy_dirs(target_dir):

    install_dir = f"{target_dir}/DANSY_DATA/"
    if not os.path.exists(install_dir):
        # Make the directory
        os.makedirs(install_dir, exist_ok=True)

    if os.path.exists(install_dir):
        # Now update the directories file
        with open(f"{DANSY_DIR}/dansy/directories.txt",'w') as f:
            f.write(install_dir)

def update_proteome_version(version):
    global DANSY_PROTEOME_VERSION 
    DANSY_PROTEOME_VERSION = version


if __name__ == '__main__':
    create_DANSy_dirs()
