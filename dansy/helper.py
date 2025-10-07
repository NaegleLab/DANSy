import os
from dansy.config import DANSY_DATA_DIR, DANSY_PROTEOME_VERSION
import dansy.ngramUtilities as ngramUtilities

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
            fullpath = os.path.join(ref_file_dir, fileName)
            all_refs.append(fullpath)
    if all_refs:
        ref_df = ngramUtilities.import_reference_file(all_refs)
    else:
        raise FileNotFoundError(f'No reference file with the suffix {ref_file_suffix} was found')
    #ref_df, interpro_dict = ngramUtilities.add_Interpro_ID_architecture(ref_df)
    
    return ref_df