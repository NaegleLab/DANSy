import time
import random
import warnings
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime
from dansy.enrichment_helpers import calculate_separation_stability, cohen_d, retrieve_fpr_checks,calculate_fpr


def calc_raw_score(dedansy,cond, seed, min_pval = -10, num_ss_trials = 100, processes = 1):
    '''
    Calculates the raw scores of the condition of interest. The raw scores are the actual network separation values, the interquartile range, and Cohen's d effect size of both of those values.

    Parameters:
    -----------
        - dedansy: deDANSy object
            The dedansy object of interest that contains all the data
        - cond: str
            The condition of interest
        - seed: int
            Seed for the random state
        - min_pval: float (Optional)
            The negative log10 value that is the smallest p-value of interest for the penalty sweep.
        - num_ss_trials: int (Optional)
            The number of subsampling and random networks to use for calculating the scores
        - process: int (Optional)
            Number of processes if multiprocessing is desired.
    
    Returns:
    --------
        - full_res: tuple
            A tuple that contains which condition, the network separation, IQR of the network separation, Cohen's d effect size, and the raw values.
    '''
    # Making the sweeping p-values 
    min_pval = abs(min_pval)
    p_vals_steps = min_pval*2 + 1
    p_vals_sweep = np.logspace(0,-min_pval, num=p_vals_steps)

    # Now adding in 0.5 and 0.05
    p_vals_sweep = np.array(sorted(np.append(p_vals_sweep, [0.5,.05]),reverse=True))
    
    random.seed(seed) # Need to reset for each of the conditions that are run to ensure reproducbility regardless of whether run with multiprocessing or not.
    hyper_sweep_res = calculate_separation_stability(dedansy, 
                                                    num_trials=num_ss_trials,
                                                    pval_sweep=p_vals_sweep,
                                                    processes=processes,
                                                    verbose=False,
                                                    )

    # Unpacking the results from the hypergeometric sweep + subsampling
    rand_ns_i = [x for x in hyper_sweep_res[0]]
    rand_iqr = [x for x in hyper_sweep_res[1]]
    subsamp_ns_i = [x for x in hyper_sweep_res[2]]
    subsample_iqr = [x for x in hyper_sweep_res[3]]

    # Now getting some of the stats and results
    iqr_res = stats.mannwhitneyu(rand_iqr,subsample_iqr)
    ns_res = stats.mannwhitneyu(rand_ns_i, subsamp_ns_i)
    iqr_d = cohen_d(subsample_iqr, rand_iqr)
    ns_d = cohen_d(subsamp_ns_i,rand_ns_i)

    # Converting to make them a little more memory efficient for saving and pandas
    subsamp_ns_i = np.array(subsamp_ns_i)
    rand_ns_i = np.array(rand_ns_i)
    subsample_iqr = np.array(subsample_iqr)
    rand_iqr = np.array(rand_iqr)

    # Now placing everything into a tuple that will be stored and converted to a DataFrame after
    full_res = (cond, ns_res, ns_d, iqr_res, iqr_d, subsamp_ns_i, rand_ns_i, subsample_iqr, rand_iqr)

    return full_res

def print_timed_message(m):
    '''
    Prints a message with the current time and date.
    '''
    y = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{y}:.....{m}")

def calculate_scores(dedansy, conds, fpr_trials=50, min_pval = -10, num_ss_trials = 100, processes = 1, seed=None, verbose=True):
    '''
    This calculates the separation and the distinct functional neighborhood scores for deDANSy given a list of conditions. The scores, Mann-Whitney U-test p-value, and FPR value are returned for each condition.

    Parameters:
    -----------
        - dedansy: deDANSy object
            The base deDANSy object containing all n-grams and expression data
        - conds: list or str
            The conditions that will be compared
        - alpha: float (Optional)
            p-value cutoff to designate DEGs
        - fc_thres: float (Optional)
            Fold change cutoff to designate DEGs
        - fpr_trials: int (Optional)
            Number of FPR trials to perform.
        - min_pval: int (Optional)
            The log10 transform of the minimum p-value to use for the p-value pruning sweep step.
        - num_ss_trials: int (Optional)
            The number of subsampled (and random) networks used to build distributions for comparing the network separation and IQR
        - processes: int (Optional)
            Number of processes to use if multiprocessing is desired. (Recommended having 4-8 when feasible)
        - seed: int
            Seed for random numbers. If not provided will use system time

    Returns:
    --------
        - dedansy_scores: pandas DataFrame
            All scores, p-values, and FPR values for each condition
        - dedansy_raw_dists: pandas DataFrame
            For each condition the raw values that made up the distributions for each score
        - dedansy_fpr_dists: pandas DataFrame
            For each condition the p-values from the random FPR trials for calculating the FPR
    '''

    # Check the min_pval parameter to make sure it is a log10 version otherwise transform it.
    if abs(min_pval) < 1:
        min_pval = np.log10(min_pval)
        warnings.warn('The provided p-value was less than 1 and thus log10 transformed.')
    
    # Check if string or list for condition(s) and convert to list if string
    if isinstance(conds, str):
        conds = [conds]
    
    # For reproducibility
    t = time.time()
    if seed is None:
        seed = int(t)
    random.seed(seed)
    seedlist = random.sample(range(1000000), len(conds)) # Will use these for the calculation trials

    if verbose:
        print_timed_message(f'Starting deDANSy scoring analysis with random seed: {seed}')

    # Now go through and get the raw scores for each condition and place results into a temporary dataframe
    full_raw_scores = []
    for i,cond in enumerate(conds):
        dedansy.calc_DEG_ngrams(cond,batch_mode=True)
        cond_res = calc_raw_score(dedansy, 
                                  cond, 
                                  seed=seedlist[i],
                                  min_pval=min_pval,
                                  num_ss_trials=num_ss_trials,
                                  processes=processes)
        full_raw_scores.append(cond_res)
        if verbose:
            print_timed_message(f'Done with enrichment-based pruning for {cond}')

    # Now process the raw scores and place them into a dataframe for output and then split off the distributions for file output if desired.  
    raw_formatted = pd.DataFrame.from_records(columns=['Comparison', 'Separation_Stats', 'Separation_Score', 'Distinction_Stats','Distinction_Score', 'Separation Subsample Dist','Separation Random Dist', 'Distinction Subsample Dist', 'Distinction Random Dist'],data=full_raw_scores)

    # Setting the indices to the comparisons so that they can be retained after
    raw_formatted.set_index('Comparison', inplace=True,drop=False)


    # Unpacking the Mann-Whitney U test results
    raw_formatted['Separation_Test_Statistic'], raw_formatted['Separation_p'] = zip(*raw_formatted['Separation_Stats'])
    raw_formatted['Distinction_Test_Statistic'], raw_formatted['Distinction_p'] = zip(*raw_formatted['Distinction_Stats'])
    raw_formatted.drop(['Separation_Stats', 'Distinction_Stats'],axis=1, inplace=True)

    # Some prep work for the FPR procedure
    ns_raw_formatted = raw_formatted.filter(['Comparison','Separation_p'],axis=1)
    iqr_raw_formatted = raw_formatted.filter(['Comparison','Distinction_p'], axis = 1)
    ns_raw_formatted.set_index('Comparison',inplace=True)
    iqr_raw_formatted.set_index('Comparison',inplace=True)

    # Splitting off the Distribution Values from the other more summarizing data
    dedansy_raw_dists = raw_formatted.filter(['Comparison','Separation Subsample Dist', 'Separation Random Dist', 'Distinction Subsample Dist', 'Distinction Random Dist'], axis =1)
    raw_formatted.drop(['Separation Subsample Dist', 'Separation Random Dist', 'Distinction Subsample Dist', 'Distinction Random Dist'], axis =1, inplace=True)

    # Making it a long format which is easier for plotting typically.
    dedansy_raw_dists = dedansy_raw_dists.melt(id_vars='Comparison').explode('value')
    
    # Now starting the False positive rate process.
    if fpr_trials > 0:
        print_timed_message('Starting the false positive rate procedure.')
        fpr_res = []
        for i,cond in enumerate(conds):
            # Reset then generate a distribution of p-values to use for the FPR calculation
            dedansy.calc_DEG_ngrams(cond,batch_mode=True)
            numDEGs = len(set(dedansy.up_DEGs).union(dedansy.down_DEGs)) # Need to use a set for instances when a gene (aka in phosphoproteomics) is shared in both conditions
            frac_up = len(dedansy.up_DEGs)/numDEGs
            internal_fpr = retrieve_fpr_checks(dedansy,
                                            numDEGs, 
                                            deg_ratios = frac_up, 
                                            processes=processes, 
                                            fpr_trials=fpr_trials,
                                            num_internal_trials=num_ss_trials,
                                            seed=seedlist[i])
            
            # Now get the p-values for the comparison in question
            a = ns_raw_formatted.loc[cond].values.tolist()[0]
            b = iqr_raw_formatted.loc[cond].values.tolist()[0]
            fprs = calculate_fpr([a,b], internal_fpr)

            # Save for export
            fpr_res.append((cond, fprs, internal_fpr))
            print_timed_message(f'Finished FPR procedure for {cond}.')
        print_timed_message('Preparing the FPR results for export')
        fpr_df = pd.DataFrame().from_records(columns=['Comparison', 'FPR Values','FPR Dists'], data =fpr_res)
        fpr_df.set_index('Comparison', inplace=True, drop=False)

        # Unpacking everything
        fpr_df['Separation_FPR'], fpr_df['Distinction_FPR'] = zip(*fpr_df['FPR Values'])
        fpr_df['NS_pval_dist'], fpr_df['IQR_pval_dist'] = zip(*fpr_df['FPR Dists'].apply(lambda x: ([i[0] for i in x], [j[1] for j in x])))

        # Now dropping the old columns
        fpr_df.drop(['FPR Values', 'FPR Dists'],axis=1, inplace=True)

        # Now splitting off the distributions and then merging the FPR values with the old results dataframe
        dedansy_fpr_dists = fpr_df.filter(['Comparison', 'NS_pval_dist', 'IQR_pval_dist'], axis =1)
        dedansy_fpr_dists = dedansy_fpr_dists.melt(id_vars='Comparison').explode('value')
        fpr_df.drop(['NS_pval_dist', 'IQR_pval_dist'],axis=1, inplace=True)
        
    else:
        print_timed_message('Skipped FPR')
        fpr_df = pd.DataFrame(columns= ['Separation_FPR','Distinction_FPR'], index=conds)
        dedansy_fpr_dists = pd.DataFrame(columns=['Comparison','variable','value'])
    
    # Merging the FPR and scores 
    raw_formatted.drop('Comparison', axis=1, inplace=True) # Dropping this as it is duplicating information in the index.
    dedansy_scores = raw_formatted.merge(fpr_df, left_index=True,right_index=True)

    # Defining the order of columns for later
    col_order = ['Comparison', 'Separation_Score','Separation_Category', 'Separation_p', 'Separation_FPR', 'Distinction_Score', 'Distinction_Category', 'Distinction_p', 'Distinction_FPR']
    
    # Now getting the score categories and absolute value of each score
    dedansy_scores['Separation_Category'] = dedansy_scores['Separation_Score'].map(lambda x: 'More' if np.sign(x) == 1 else 'Less')
    dedansy_scores['Distinction_Category'] = dedansy_scores['Distinction_Score'].map(lambda x: 'Stably Distinct' if np.sign(x) == 1 else 'Unstable/Overlapping')
    dedansy_scores['Separation_Score'] = np.abs(dedansy_scores['Separation_Score'])
    dedansy_scores['Distinction_Score'] = np.abs(dedansy_scores['Distinction_Score'])


    dedansy_scores = dedansy_scores.filter(items=col_order, axis=1)

    return dedansy_scores, dedansy_raw_dists, dedansy_fpr_dists