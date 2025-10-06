import json
import os
import pandas as pd
import numpy as np
import dansy.ngramUtilities as ngramUtilities

# Global variables for the complete proteome on what default values will look like.
# Note for memory efficiency we save the adjacency as a json file as it is a sparse matrix.
DANSY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANSY_DATA_DIR = DANSY_DIR + '/DANSY_DATA'
DANSY_ADJ_FILE = 'DANSy_Complete_Proteome_Adjacency.json' 
DANSY_COLLAPSED_NGRAMS_FILE = 'DANSy_Complete_Proteome_Collapsed_ngrams.csv'
DANSY_PROTEOME_VERSION = '20250512.csv'

def main():
    check_dirs()

    full_adj_path = os.path.join(DANSY_DATA_DIR, DANSY_ADJ_FILE)
    col_ngram_path = os.path.join(DANSY_DATA_DIR, DANSY_COLLAPSED_NGRAMS_FILE)

    if os.path.exists(full_adj_path) and os.path.exists(col_ngram_path):
        print('DANSy data files for the complete proteome already exist.')
    else:
        print('Generating the complete proteome adjacency analysis.')
        _,_ = generate_complete_adjacency()

def check_dirs():
    if not os.path.exists(DANSY_DATA_DIR):
        os.makedirs(DANSY_DATA_DIR, exist_ok=True)

def import_proteome_data(adj_file = DANSY_ADJ_FILE, adj_dir = DANSY_DATA_DIR,col_ngrams_file = DANSY_COLLAPSED_NGRAMS_FILE, colngram_dir = DANSY_DATA_DIR):
    '''
    This imports both the adjacency matrix and collapsed n-grams from DANSy for the complete proteome.

    Parameters
    ----------
        - adj_file: str
            Name of the file containing the adjacency data (should be a json file)
        - adj_dir: str
            Directory of where the adjacency file is located
        - col_ngrams_file: str
            Name of the txt file containing the collapsed n-grams information
        - colngram_dir: str
            Directory where the collapsed n-gram file is located.

    Returns
    -------
        - adj: pandas DataFrame
            Dataframe containing the adjacency matrix of the complete proteome
        - collapsed_ngrams: list
            List of the collapsed n-grams from the DANSy analysis
    '''
    adj = import_adjacency(adj_file=adj_file, adj_dir=adj_dir)
    collapsed_ngrams = import_collapsed_ngrams(col_ngrams_file=col_ngrams_file,
                                               col_ngram_dir = colngram_dir)
    
    return adj, collapsed_ngrams

def generate_complete_adjacency(ref_dir = None,ref_version = None, target_dir = None):
    '''
    Base function that conducts the full n-gram analysis pipeline on the complete canonical human proteome and generates the adjacency matrix and the n-grams that were collapsed due to redundant information.

    Parameters:
    -----------
        - generate_files: bool
            Flag to determine if csv files should be generated in the project directory
        - readable_flag: bool
            Flag to determine if the adjacency matrix should use the InterPro IDs or the domain names
        - filename: list
            Strings for the name of files to be generated.
        - ref_version: str
            Version of reference files to use indicated by the suffix of the file name
        -ref_dir: str
            Directory for the csv file
    
    Returns:
    --------
        - full_adj: pandas DataFrame
            dataframe that contains the adjacency matrix of the complete proteome. Self-loops have been removed
        - collapsed_ngrams: list
            List of the n-grams that had redundant information that were collapsed during analysis.
    '''

    # If a directory is not provided defaulting to creating/writing to a DANSy specific directory
    if target_dir == None:
        target_dir = DANSY_DATA_DIR
        check_dirs()

    if ref_dir == None:
        ref_dir == DANSY_DATA_DIR

    ref_df, interpro_dict = import_proteome_files(ref_version, ref_dir)
    all_domains = [x for x in interpro_dict.keys()]
    adj_df, _, _, collapsed_ngrams, _ = ngramUtilities.full_ngram_analysis(ref_df, all_domains, min_arch=1, readable_flag=False, max_ngram=67,max_node_len=67)

    full_adj = adj_df.copy()
    for node in full_adj.columns:
        full_adj.loc[node,node]=0
   
    # Now save the files for later loading
    generate_adjacency_json(full_adj)
    col_fullpath = os.path.join(target_dir, DANSY_COLLAPSED_NGRAMS_FILE)
    np.savetxt(col_fullpath, collapsed_ngrams, delimiter = ',', fmt = '%s')
    return full_adj, collapsed_ngrams

def import_proteome_files(ref_file_dir = DANSY_DATA_DIR, ref_file_suffix = DANSY_PROTEOME_VERSION):
    '''
    Imports the files that are used for the generation of the reference dataframe of the complete canonical proteome.

    Note: Need to adjust this so it looks in only one folder from here on out.

    Parameters:
    -----------
        - reference_file_version: str
          String of the suffix of the reference files to be used  

    Returns:
    --------
        - ref_df: pandas DataFrame
            Dataframe containing the InterPro, UniProt, and PDB information of individual proteins as retrieved via CoDIAC
        - interpro_dict: dict
            dictionary containing the InterPro IDs and domain names for conversion purposes
    
    '''
    all_refs = []
    
    ref_files = os.listdir(ref_file_dir)
    for fileName in ref_files:
        if fileName.endswith(ref_file_suffix):
            fullname = os.path.join(ref_file_dir, fileName)
            all_refs.append(fullname)

    ref_df = ngramUtilities.import_reference_file(all_refs)
    ref_df, interpro_dict = ngramUtilities.add_Interpro_ID_architecture(ref_df)

    # If the full proteome reference files already had Interpro IDs this can lead to an empty dictionary
    if len(interpro_dict) == 0:
        interpro_dict = ngramUtilities.generate_interpro_conversion(ref_df)

    return ref_df, interpro_dict

def import_adjacency(adj_file = DANSY_ADJ_FILE, adj_dir = DANSY_DATA_DIR):
    '''
    Imports the adjacency matrix from either a designated file or by generating a new file. 

    Parameters:
    -----------
        adj_file: str
            File name of the adjacency file
        adj_dir: str (optional)
            Directory Name of where the adjacency file is located. By default it will place files in the data folder.
    
    Returns:
    --------
        full_adj: pandas DataFrame
            Full adjacency matrix of the n-gram analysis of the human proteome.

    '''

    full_path = os.path.join(adj_dir, adj_file)
    if os.path.exists(full_path):
        js_adj = pd.read_json(full_path, orient = 'index')
        js_adj.fillna(0, inplace=True)
        full_adj = js_adj.astype('int64')
    else:
        print('Generating a new Adjacency Matrix but not a csv file.')
        full_adj, _ = generate_complete_adjacency()

    return full_adj

def import_collapsed_ngrams(col_ngrams_file = DANSY_COLLAPSED_NGRAMS_FILE, col_ngram_dir = DANSY_DATA_DIR):
    '''
    Imports the list of n-grams that were removed during n-gram analysis of the complete proteome from either the file or via the n-gram analysis.

    Parameters:
    -----------
        rm_ngrams_file: str (optional)
            File name if not provided will generate a File with the title: Current_Complete_Proteoeme_collapsed_ngrams.csv
        rm_dir: str (optional)
            Directory Name of where the file is located. By default it will place files in the data folder.
    
    Returns:
    --------
        collapsed_ngrams: list
            List of subsumed n-grams with redundant information

    '''

    full_path = os.path.join(col_ngram_dir, col_ngrams_file)
    if os.path.exists(full_path):
        temp = pd.read_csv(full_path, header = None)
        collapsed_ngrams = temp[0].tolist()

    else:
        _, collapsed_ngrams = generate_complete_adjacency()

    return collapsed_ngrams

def generate_adjacency_json(base_adj, adj_file = DANSY_ADJ_FILE, adj_dir = DANSY_DATA_DIR):
    '''
    Imports the adjacency matrix from the designated json file. The JSON file should have keys correspond to the index of the final adjacency dataframe.
    
    Parameters
    -----------
        adj_file: str
            File name for the csv file corresponding to the adjacency matrix.
        adj_dir: str (optional)
            Directory Name of where the adjacency file is located. By default it will look for files in the data folder.
    
    Returns
    --------
        adj: pandas DataFrame
            Full adjacency matrix of the n-gram analysis of the human proteome.
    '''

    adj_dict = base_adj.to_dict(orient='index')

    # Running through each entry and removing the 0s except the self-referencing ones to preserve adjacency matrix shape and reduce memory footprint.
    for ngram in adj_dict.keys():
    
        keys_2_rm = base_adj.index[base_adj.loc[ngram] == 0].tolist()
        for k in keys_2_rm:
            if k != ngram:
                del adj_dict[ngram][k]
    
    fullpath = os.path.join(adj_dir, adj_file)
    with open(fullpath,'w') as output:
        json.dump(adj_dict, output)
    
    return None

if __name__ == '__main__':
    main()