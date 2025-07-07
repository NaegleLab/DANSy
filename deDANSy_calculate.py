import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pybiomart import Dataset
import seaborn as sns
import random
from tqdm import tqdm
import scipy.stats as stats
import time
from datetime import datetime
import dansy
from dansy.enrichment_helpers import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="File containing the DEG information")
    parser.add_argument("output", help="Prefix for the files that will be generated for different sets of results.")
    parser.add_argument("comparisons", nargs="*", help="List of comparisons that are to be made for the analysis.")
    parser.add_argument("-d","--data-id", help="String of the data column containing the IDs for data conversion.", required=False, default='ensembl_gene_id')
    parser.add_argument("-mp", "--multiprocess", help = "The number of processes to run for longer network separation calculations.", default=1, type=int)
    parser.add_argument("-sN","--subsampleN", help="Number of trials to do for subsampling procedure.", default=50, type=int)
    parser.add_argument("-fN","--fprN", help="Number of trials to do for calculating the false positive rate.", default=50, type=int)
    parser.add_argument('-a','--alpha', help='p-value threshold for determining DEGs', default=0.05, type=float)
    parser.add_argument('-fc','--fcthres', help='Fold change threshold for designating DEGs', default=1, type=float)
    parser.add_argument("--seed", help="Random number generator seed.", default=None, type=int)

    # Setting up the user inputs
    args = parser.parse_args()
    filename = args.filename
    output_prefix = args.output
    conds = args.comparisons
    data_id = args.data_id
    processes = args.multiprocess
    seed = args.seed
    num_ss_trials = args.subsampleN
    fpr_trials = args.fprN
    alpha = args.alpha
    fcthres = args.fcthres
   
    # Setting the random seed for reproducibility
    t = time.time()
    if seed == None:
        seed = int(t)
    random.seed(seed)
    seedlist = random.sample(range(1000000), len(conds))
    print_timed_message(f"Seed for the analysis was {seed}")

    #### Setting up the reference data
    print_timed_message('Importing the Reference')
    # Importing the entire proteome
    complete_ref,_ = dansy.import_proteome_files(ref_file_dir = 'data/Current_Human_Proteome/')

    #### pybiomart database for gene name conversions
    print_timed_message('Getting the ID conversions dataset')
    dataset = Dataset(host = 'http://useast.ensembl.org', name='hsapiens_gene_ensembl',)
    gene_ID_conv = dataset.query(attributes=['ensembl_gene_id','external_gene_name','external_synonym','uniprotswissprot'])

    ##### Now importing the dataset
    print_timed_message('Importing the dataset')
    deg_dataset = pd.read_csv(filename)
    degnn = dansy.deDANSy(deg_dataset, gene_ID_conv,uniprot_ref=complete_ref, data_ids=data_id)

    #### Conducting the analysis
    print_timed_message('Now getting the baseline network separation sweeps')
    p_vals_sweep = np.logspace(0,-10, num=21)
    p_vals_sweep = np.array(sorted(np.append(p_vals_sweep, [0.5,.05]),reverse=True))
    full_res = []
    
    for i,cond in enumerate(conds):
        random.seed(seedlist[i]) # Need to reset for each of the conditions that are run to ensure reproducbility regardless of whether run with multiprocessing or not.
        degnn.calc_DEG_ngrams(data_cols=['log2FoldChange_'+cond,'padj_'+cond],alpha=alpha,fc_thres=fcthres, batch_mode=True)
        numDEGs = len(degnn.up_DEGs) + len(degnn.down_DEGs)
        full_ns = degnn.DEG_network_sep(force_run=True)
        hyper_sweep_res = calculate_separation_stability(degnn, 
                                                        num_trials=num_ss_trials,
                                                        pval_sweep=p_vals_sweep,
                                                        processes=processes,
                                                        verbose=False,
                                                        progress_bar=False)
    
        # Unpacking the results from the hypergeometric sweep + subsampling
        rand_full_ns = [x for x in hyper_sweep_res[0]]
        rand_iqr = [x for x in hyper_sweep_res[1]]
        actual_full_ns = [x for x in hyper_sweep_res[2]]
        subsample_iqr = [x for x in hyper_sweep_res[3]]
        
        # Now getting some of the stats and results
        iqr_res = stats.mannwhitneyu(rand_iqr,subsample_iqr)
        ns_res = stats.mannwhitneyu(rand_full_ns, actual_full_ns)
        iqr_d = cohen_d(subsample_iqr, rand_iqr)
        ns_d = cohen_d(actual_full_ns,rand_full_ns)

        # Converting to make them a little more memory efficient for saving and pandas
        actual_full_ns = np.array(actual_full_ns)
        rand_full_ns = np.array(rand_full_ns)
        subsample_iqr = np.array(subsample_iqr)
        rand_iqr = np.array(rand_iqr)

        # Now placing everything into a tuple that will be stored and converted to a DataFrame after
        full_res.append((cond,full_ns, ns_res, ns_d, iqr_res, iqr_d, actual_full_ns, rand_full_ns, subsample_iqr, rand_iqr))
        print_timed_message(f'Done with the sweep for {cond}')


    print_timed_message('Prepping the subsampled hypergeometric data for export.')
    # Now converting the full list to a Dataframe
    res_df = pd.DataFrame.from_records(columns=['Comparison', 'Network Separation', 'NS Stats', 'NS Cohen', 'IQR Stats','IQR Cohen', 'NS Subsample Dist','NS Random Dist', 'IQR Subsample Dist', 'IQR Random Dist'],data=full_res)

    # Unpacking the Mann-Whitney U test results
    res_df['NS Statistic'], res_df['NS p'] = zip(*res_df['NS Stats'])
    res_df['IQR Statistic'], res_df['IQR p'] = zip(*res_df['IQR Stats'])
    res_df.drop(['NS Stats', 'IQR Stats'],axis=1, inplace=True)

    # Transforming the p-values to -log10 to make it easier to plot
    res_df['IQR p_log'] = -np.log10(res_df['IQR p'])
    res_df['NS p_log'] = -np.log10(res_df['NS p'])

    # Some prep work for the FPR procedure
    ns_res_df = res_df.filter(['Comparison','NS p'],axis=1)
    iqr_res_df = res_df.filter(['Comparison','IQR p'], axis = 1)
    ns_res_df.set_index('Comparison',inplace=True)
    iqr_res_df.set_index('Comparison',inplace=True)

    # Splitting off the Distribution Values from the other more summarizing data
    distrib_res_df = res_df.filter(['Comparison','NS Subsample Dist', 'NS Random Dist', 'IQR Subsample Dist', 'IQR Random Dist'], axis =1)
    res_df.drop(['NS Subsample Dist', 'NS Random Dist', 'IQR Subsample Dist', 'IQR Random Dist'], axis =1, inplace=True)

    # Now prepping the two dataframes for export
    distrib_res_df = distrib_res_df.melt(id_vars='Comparison').explode('value')
    

    # Here is saving the data that will then be imported later if the process is run again for plotting purposes only
    distrib_res_df.to_csv(output_prefix+'_Subsample_Results_Underlying_Values.csv', index=False)

    print_timed_message('Successfully exported the subsampled hypergeometric results.')

    # Now starting the False positive rate process.
    print_timed_message('Starting the false positive rate procedure.')
    fpr_res = []
    for i,cond in enumerate(conds):
        
        # Reset then generate a distribution of p-values to use for the FPR calculation
        degnn.calc_DEG_ngrams(data_cols=['log2FoldChange_'+cond,'padj_'+cond],alpha=alpha,fc_thres=fcthres,batch_mode=True)
        numDEGs = len(degnn.up_DEGs) + len(degnn.down_DEGs)
        frac_up = len(degnn.up_DEGs)/numDEGs
        internal_fpr = retrieve_fpr_checks(degnn,
                                        numDEGs, 
                                        deg_ratios = frac_up, 
                                        processes=processes, 
                                        fpr_trials=fpr_trials,
                                        num_internal_trials=num_ss_trials,
                                        seed=seedlist[i])
        
        # Now get the p-values for the comparison in question
        a = ns_res_df.loc[cond].values.tolist()[0]
        b = iqr_res_df.loc[cond].values.tolist()[0]
        fprs = calculate_fpr([a,b], internal_fpr)

        # Save for export
        fpr_res.append((cond, fprs, internal_fpr))
       
        print_timed_message(f'Done with calculating the FPR for {cond}')

    print_timed_message('Preparing the FPR results for export')
    fpr_df = pd.DataFrame().from_records(columns=['Comparison', 'FPR Values','FPR Dists'],data =fpr_res)

    # Unpacking everything
    fpr_df['NS_FPR'], fpr_df['IQR_FPR'] = zip(*fpr_df['FPR Values'])
    fpr_df['NS_pval_dist'], fpr_df['IQR_pval_dist'] = zip(*fpr_df['FPR Dists'].apply(lambda x: ([i[0] for i in x], [j[1] for j in x])))

    # Now dropping the old columns
    fpr_df.drop(['FPR Values', 'FPR Dists'],axis=1, inplace=True)

    # Now splitting off the distributions and then merging the FPR values with the old results dataframe
    fpr_dists_df = fpr_df.filter(['Comparison', 'NS_pval_dist', 'IQR_pval_dist'], axis =1)
    fpr_dists_df = fpr_dists_df.melt(id_vars='Comparison').explode('value')
    fpr_df.drop(['NS_pval_dist', 'IQR_pval_dist'],axis=1, inplace=True)
    res_df = res_df.merge(fpr_df)

    fpr_dists_df.to_csv(output_prefix+'_FPR_Value_Distributions.csv', index=False)
    res_df = res_df.melt(id_vars='Comparison')
    res_df.to_csv(output_prefix+'_Results_Summary_Stats.csv',index=False)
    print_timed_message('Succesfully exported the results.')

def print_timed_message(m):
    y = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{y}:\t{m}")

if __name__ == '__main__':
    main()