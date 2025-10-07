import dansy
import os
import pandas as pd

def main():

    # Checking that the config module works as intended
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dansy.config.create_DANSy_dirs(cwd)
    assert os.path.exists(dansy.DANSY_DATA_DIR), "Issue with creating the data directory"

    # Check that updating the proteome can be imported
    ref = dansy.helper.import_proteome_files()
    assert isinstance(ref, pd.DataFrame), "Issue with the reference file"

    original_version = dansy.config.DANSY_PROTEOME_VERSION
    assert original_version == '20250512.csv', "Leftover version update"

    # Try to update the ref and make sure we get an error when trying to import an improper version
    new_version = '20251010.csv'
    try:
        new_ref = dansy.helper.import_proteome_files(ref_file_suffix=new_version)
    except FileNotFoundError:
        new_ref = pd.DataFrame()
    
    assert not ref.equals(new_ref), "Issue with importing a version that shouldn't work"

    dansy.config.update_proteome_version(new_version, default=False)

    version_file = f"{dansy.config.DANSY_DIR}/dansy/current_proteome_version.txt"
    prot_ver = []
    with open(version_file, 'r') as f:
        for line in f:
            prot_ver.append(line)
    current_version = prot_ver[0]
    assert current_version == original_version, "The default value was updated when it was not intended."
    assert dansy.config.DANSY_PROTEOME_VERSION == new_version, "Proteome version did not get updated."

    dansy.config.update_proteome_version(original_version, default=True)

    # Now let's make sure we can still create a DANSy object from a small subset without providing the reference file
    test_ref = pd.read_csv('tests/test_data/small_tyrkin_reference.csv')
    protsOI = test_ref['UniProt ID'].tolist()
    wp = dansy.dansy(protsOI=protsOI, n=10)

    assert wp.protsOI == protsOI, "Issue with proteins of interest."
    # Since this uses the reference file from the whole proteome it should be just slightly different (mostly the indices) than our smaller version.
    assert not wp.ref.equals(test_ref), "The reference file that was imported for the dansy object was identical to a smaller reference file"

if __name__ == '__main__':
    main()
