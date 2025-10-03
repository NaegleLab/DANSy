import dansy
import pandas as pd
import os
import networkx as nx

def main():
    ref_file_path = 'tests/test_data'
    filename = 'small_tyrkin_reference.csv'
    ref_file = os.path.join(ref_file_path, filename)
    ref_df = pd.read_csv(ref_file)

    # Checking a collapsed network
    d = dansy.dansy(ref = ref_df, n = 10, verbose=False)
    ngram_check = set(d.collapsed_ngrams).intersection(d.ngrams)
    
    # Simple checks to verify the n-gram network is being properly executed and can be passed to networkx for analysis.
    assert len(d.ngrams) == 20, "The wrong number of n-grams were extracted."
    assert len(ngram_check) == 0, "There was overlap between the collapsed n-gram nodes and the final n-grams in the DANSy network."
    assert d.G.degree('IPR000719') == 15, "The edges connected to kinase domain is not correct."
    assert nx.betweenness_centrality(d.G), "Getting the network centrality metrics is not working properly."

    # Checking the uncollapsed version is working properly
    d = dansy.dansy(ref=ref_df, n=10, collapse = False, verbose=False)
    assert len(d.collapsed_ngrams) == 0, "Network collapsed n-grams when it shouldn not have."

    # Making sure that it properly extracts n-gram lengths as intended
    d = dansy.dansy(ref=ref_df, n=1, verbose=False)
    ngram_lengths = list({len(i.split('|')) for i in d.ngrams})
    assert len(ngram_lengths) == 1, "When extracting n-grams of length one it extracted multiple lengths."
    assert ngram_lengths[0] == 1, "The wrong n-gram lengths were extracted."

    d = dansy.dansy(ref=ref_df, n=3, verbose=False)
    ngram_lengths = list({len(i.split('|')) for i in d.ngrams})
    assert len(ngram_lengths) >= 1, "When extracting n-grams of length 3 too few n-gram lengths were extracted."
    assert max(ngram_lengths) == 3, "The wrong n-gram lengths were extracted."

    # Check the error catching
    bad_ref = ref_df.filter(['UniProt ID','Gene'])
    try:
        d = dansy.dansy(ref=bad_ref, verbose=False)
        error_flag = False
    except dansy.DomainArchError:
        error_flag = True
    else:
        error_flag = False

    assert error_flag, "Domain architecture format erroring check failed."


if __name__ == '__main__':
    main()