import dansy
import pandas as pd
import numpy as np
import os

def main():

    data_path = 'tests/test_data'
    deg_file = 'DEG_test_data.csv'
    ref_file = 'small_tyrkin_reference.csv'
    conv_file = 'ID_conversion.csv'

    ex_deg_data = pd.read_csv(os.path.join(data_path, deg_file))
    ref_df = pd.read_csv(os.path.join(data_path, ref_file))
    conv_id = pd.read_csv(os.path.join(data_path, conv_file))

    d = dansy.DEdansy(dataset=ex_deg_data,
                      id_conv=conv_id,
                      conv_cols='Gene stable ID',
                      data_ids='Gene stable ID',
                      run_conversion=True,
                      uniprot_ref=ref_df)
    
    # Creating the contrasting metadata
    comps = ['simple_fc','fc1_p0.05','fc1_p0.01']
    d.create_contrast_metadata(comparisons=comps,
                               fc_cols=['log2FC']*3,
                               pval_cols=['pval']*3,
                               fcs=[0.5,1,1],
                               alphas=[1,0.05,0.01])
    
    # Now going through and calculating the n-grams and testing simple functions associated with them
    d.calc_DEG_ngrams(comp = 'simple_fc')
    assert len(d.up_ngrams) > 0, 'Calculating enriched n-grams failed for the most permissive cutoff'
    assert not np.isnan(d.DEG_network_sep()), 'Network separation for the permissive cutoffs failed.'

    try:
        d.plot_DEG_ns()
        plot_flag = True
    except:
        plot_flag = False
    assert plot_flag, 'Plotting the deDANSy network failed.'

    # Check the condition that is supposed to give a nan because it has only one n-gram that is considered up/down regulated and none for the other condition.
    d.calc_DEG_ngrams(comp = 'fc1_p0.01')
    assert np.isnan(d.DEG_network_sep()), 'Failed to return the proper NaN for the restrictive cutoff method.'
    # Now testing the enrichment and deDANSy score methods that they function properly
    for i in comps:
        assert test_enrichment(d,i), f"Failed to calculate n-gram enrichment for {i}."

    try: 
        d.calculate_scores(comps=comps, fpr_trials=2,min_pval=-3, num_ss_trials=10, verbose=False)
        score_flag = True
    except:
        score_flag = False
    assert score_flag, "Failed to calculate deDANSy scores."

    # Check the plotting function
    try:
        d.plot_scores()
        plot_flag = True
    except:
        plot_flag = False
    assert plot_flag, 'Plotting the deDANSy scores failed.'


def test_enrichment(dansy_obj:dansy.DEdansy, comp: str):
    ''' Tests if the n-gram enrichment runs properly and does not error out'''

    try:
        dansy_obj.calculate_ngram_enrichment(comparison=comp,fpr_trials=5)
        enrich_flag = True
    except:
        enrich_flag = False

    return enrich_flag

if __name__ == '__main__':
    main()

