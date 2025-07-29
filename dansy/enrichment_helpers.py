import random
import numpy as np
import scipy.stats as stats
import multiprocessing as mp
import dansy.network_separation_helpers as ns_helpers

# Defining a function to calculate cohen's D since I do not want to make the assumption that there is equal variance between the random and subsampled distributions. 
def cohen_d(a,b):
    '''
    Calculates the Cohen's d effect size of 2 lists of values, where the second is considered the "control" group.

    Parameters:
    -----------
        a,b: list or numpy array
            Distribution of values to compare the effect size of
    
    Returns:
    --------
        d: numpy float
            Cohen's d effect size (Returns nan if all values are nan)

    '''
    
    # For certain cases there might be all nans so just check that first
    if all(np.isnan(a)) and all(np.isnan(b)): 
        d = np.nan
    else:
        # Means
        x1 = np.nanmean(a)
        x2 = np.nanmean(b)

        # Sizes
        n1 = len(a)
        n2 = len(b)

        # Defining the pooled standard deviation (Note: need to account for the definition by cohen that uses N-1 for the variance, which numpy does not use)
        s = np.sqrt(((n1-1)*np.nanstd(a,ddof=1)**2 + (n2 -1)*np.nanstd(b, ddof=1)**2)/(n1+n2-2))
        if s > 0:
            d = (x1-x2)/s
        elif np.isnan(s):
            d = np.nan
        else: #Shouldn't ever get to this
            d = 0
        

    return d

def hypergeom_prune_ns(dedansy, sweep):
    '''
    Builds a distribution of network separation values between two conditions on the deDANSy network based on the enrichment pruning over more restrictive p-value thresholds.

    Parameters:
    ----------- 
        - dedansy: deDANSy object
            The object containing all the enriched n-grams for both conditions of interest
        - sweep: list
            List of thresholds to use for pruning of differential expression status

    Returns:
    --------
        - ns_sweep: list
            Network separation values for each of the thresholds provided
    '''

    # Get the enrichment p-values for each n-gram of both conditions
    up_hyper_vals = dedansy.ngram_DEG_hypergeom('Up')
    dn_hyper_vals = dedansy.ngram_DEG_hypergeom('Down')
    
    ns_sweep = []
    for i in sweep:
        
        # Get only n-grams that pass the threshold
        up_check = [k for k,v in up_hyper_vals.items() if v <= i]
        dn_check = [k for k,v in dn_hyper_vals.items() if v <= i]

        # Setting up the networks for determining the network separation
        up_net = dedansy.G.subgraph(up_check)
        dn_net = dedansy.G.subgraph(dn_check)
        if len(up_net.nodes()) > 0 and len(dn_net.nodes())> 0:
            ns = ns_helpers.network_separation(up_net, dn_net, dedansy.ref_data)
        else:
            ns = np.nan    
        ns_sweep.append(ns)

    return ns_sweep

# Below is a handful of helper functions to try and do network separation on a number of iterations to create a null distribution of DEG network separation values
def get_random_count_dist(count_weights):
    '''
    Creates list of counts for different domain architecture lengths to use in subsampling and random gene selection to calculate robust enrichment of different n-grams. This creates a small bit of randomness to ensure some leeway of genes chosen to not bias n-grams of specific lengths being consistently over-represented. The total distribution of domain architectures from 0 to 10 are individual elements, while those >10 will be accumulated as these represent <5% of the human proteome.

    Parameters:
    -----------
        - count_weights: list
            List of counts of the total distribution of domain architecture lengths of the measured universe in the deDANSy object
    
    Returns:
    --------
        - rand_counts: list
            List of random integers for each domain architecture length
    '''
    rand_counts = []
    for i in count_weights[1:11]: #Skip the 0 length architectures as they do not contribute to the network
        
        # Give leeway of about +/-15% for randomly choosen an integer of similar size to the original count
        i_min = np.floor(i*.85)
        i_max = np.ceil(i*1.15)
        if i_min == 0 and i_max == 0:
            r = 0
        else:
            r = random.randrange(i_min, i_max)
        rand_counts.append(r)
    
    # For the 10+ architectures choose between 0 and the actual value since going above will really skew n-grams
    if count_weights[11] > 0:
        rand_counts.append(random.randrange(0,count_weights[11]))
    else:
        rand_counts.append(0)

    return rand_counts

def get_random_id_indices(arch_len_array, rand_count_list):
    '''
    Get the indices of randomly chosen genes to use as a null distribution.

    Parameters:
    -----------
        - arch_len_array: list
            The list of protein architecture lengths to choose randomly from
        - rand_count_list: list
            The number of genes to choose with each architecture length
    
            
    Returns:
    --------
        - rand_4_analysis: list
            List of indices from the original dataframe containing all protein information.
    '''
    rand_4_analysis = []
    for i in range(1,11):
        cands = arch_len_array[arch_len_array == i]
        rand_ids = random.sample(list(cands.index), k=rand_count_list[i-1]) # Using indices as these are conserved across all the analysis and have smallish memory footprint
        rand_4_analysis += rand_ids

    # For the >10 n-grams now getting candidates
    cands = arch_len_array[arch_len_array > 10]
    rand_ids = random.sample(list(cands.index), k=rand_count_list[10])
    rand_4_analysis += rand_ids

    return rand_4_analysis

def designate_rand_DEGs(rand_indices, ref_df, up_fraction):
    '''
    Designates randomly chosen genes as either up or down regulated while preserving the relative fraction of each condition.

    Parameters:
    -----------
        - rand_indices: list
            List of protein indices from the reference data frame that were randomly chosen
        - ref_df: pandas DataFrame 
            The reference dataframe that contains all protein info
        - up_fraction: float
            The relative fraction of up to down regulated genes/proteins being expressed
    
    Returns:
    --------
        - rand_DEGs: dict
            Up and down lists of UniProt IDs from the randomly chosen genes
    '''

    # Now double-checking the distribution
    random_prots = ref_df.loc[rand_indices]

    # Now designating the random genes as either up or down based on the proportion of DEGs that were for both
    rand_up_num = round(up_fraction*len(random_prots))

    # Alternative to the random sampling which has issues with reproducibility when called multiple times
    rand_prot_set = sorted(random_prots['UniProt ID'].tolist())
    #random.shuffle(rand_prot_set)
    #rand_up_chosen = rand_prot_set[0:rand_up_num]
    #rand_down_chosen = rand_prot_set[rand_up_num:]

    rand_up_chosen = sorted(random.sample(rand_prot_set, k=rand_up_num))
    rand_down_chosen = sorted(set(rand_prot_set).difference(rand_up_chosen))
    rand_DEGs = {'Up':rand_up_chosen,'Down':rand_down_chosen}
    return rand_DEGs

def get_random_net_sep_metrics(normalized_arch_weights, complete_arch_dist, dedansy,deg_ratio,pvals= np.logspace(0,-10, num=21),return_dist = False):
    '''
    Performs the network separation and pruning analysis on randomly chosen genes from the dedansy object.

    Parameters:
    -----------
        - normalized_arch_weights: list
            The relative fraction of the domain architecture lengths from 0-10 and 10+ from the deDANSy object of differentially expressed genes/proteins
        - complete_arch_dist: list
            The proteins that are found for each domain architecture length to choose gene randomly from
        - dedansy: deDANSy object
            The object that contains all the protein information being analyzed
        - deg_ratio: float
            The relative fraction of up to down regulated genes/proteins
        - pvals: numpy array (Optional)
            The threshold p-values to use for pruning analysis
        - return_dist: bool (Optional)
            Whether the distribution of p-values during the p-value sweep is return. Default is False as this is a niche request

    Returns:
    --------
        - o: tuple
            Tuple containing both the network separation prior to pruning and the interquartile range of network separation values during the pruning analysis
        
    '''
    
    # Setting the randomly chosen genes
    rcd = get_random_count_dist(normalized_arch_weights)
    rii = get_random_id_indices(complete_arch_dist,rcd)
    r_prots = designate_rand_DEGs(rii, dedansy.ref,deg_ratio)
    dedansy.set_DEG_ngrams(r_prots['Up'],r_prots['Down'], verbose=False)

    # Getting the two (or three) metrics
    ns = dedansy.DEG_network_sep()
    ns_vals = hypergeom_prune_ns(dedansy, pvals)
    ns_iqr = stats.iqr(ns_vals, nan_policy='omit')
    if return_dist:
        o = (ns,ns_iqr,ns_vals)
    else:
        o = (ns,ns_iqr)

    return o

def get_subsample_net_sep_metrics(dedansy,orig_DEGs, rand_prot_nums,pvals= np.logspace(0,-10, num=21),return_dist = False):
    '''
    Performing the network separation and pruning analysis on a subsampled set of proteins from the differentially expressed genes/proteins.

    Parameters:
        - dedansy: deDANSy object
            The dansy object containing all expression and protein information
        - orig_DEGs: list
            List of two elements which contain lists of all the UniProt IDs that are either up or down (in that order) regulated
        - rand_prot_nums: list
            Number of randomly chosen proteins for each condition to ensure we match the randomly chosen genes
        - pvals: numpy array
            The threshold p-values to use during the pruning step
        - return_dist: bool
            Whether the distribution of p-values during the p-value sweep is return. Default is False as this is a niche request

    Returns:
    --------
        - o: tuple
            Tuple containing both the network separation prior to pruning and the interquartile range of network separation values during the pruning analysis
        
    '''
    
    # Setting up the original differentially expressed genes/proteins and randomly sampling them
    orig_up = orig_DEGs[0]
    orig_dn = orig_DEGs[1]
    rand_up = random.sample(orig_up, k=rand_prot_nums[0])
    rand_dn = random.sample(orig_dn, k=rand_prot_nums[1])

    # Setting the subsampled genes and getting the metrics of interest
    dedansy.set_DEG_ngrams(rand_up,rand_dn, verbose=False)
    ns = dedansy.DEG_network_sep()
    ns_vals = hypergeom_prune_ns(dedansy,pvals)
    ns_iqr = stats.iqr(ns_vals, nan_policy='omit')

    if return_dist:
        o = (ns,ns_iqr,ns_vals)
    else:
        o = (ns,ns_iqr)

    return o

def individual_trial_calc(dedansy, arch_weights, comp_arch_dist, ratio, sweep, originals,dist_flag = False, seed =123):
    '''
    Calculates the subsampled and randomly chosen genes network separtion and pruning analysis.

    Parameters:
    -----------
        - dedansy: deDANSy object
            The dansy object containing all expression and protein information
        - arch_weights: list
            The relative fraction of the domain architecture lengths from 0-10 and 10+ from the deDANSy object of differentially expressed genes/proteins
        - comp_arch_dist: list
            The proteins that are found for each domain architecture length to choose gene randomly from
        - ratio: float
            The relative fraction of up to down regulated genes/proteins
        - sweep: numpy array
            The threshold p-values to use for pruning analysis
        - originals: list
            List of two elements which contain lists of all the UniProt IDs that are either up or down (in that order) regulated
        - dist_flag: bool (Optional)
            Whether the distribution of p-values during the p-value sweep is return. Default is False as this is a niche request
        - seed: int (Optional)
            Random number generator seed to ensure reproducibility. Default is 123
    
    Returns:
    --------
        - list of tuples
            The random and then subsampled data of the unpruned network separation, the interquartile range of network separation during pruning, and if requested the distribution of values
    '''

    random.seed(seed)
    rand_data = get_random_net_sep_metrics(normalized_arch_weights=arch_weights,
                                          complete_arch_dist=comp_arch_dist,
                                          dedansy=dedansy,
                                          deg_ratio=ratio,
                                          pvals=sweep,
                                          return_dist=dist_flag)
    # Since the random function above generates the sizes will pass this on.
    rand_szs = (len(dedansy.up_DEGs), len(dedansy.down_DEGs))
    subsample_data = get_subsample_net_sep_metrics(dedansy=dedansy,
                                orig_DEGs=originals,
                                rand_prot_nums=rand_szs,
                                pvals=sweep,
                                return_dist=dist_flag)
    
    return list(rand_data + subsample_data)
    
def calculate_separation_stability(dedansy, num_trials = 50, pval_sweep = np.logspace(0,-10,21), return_distributions = False,processes = 1, verbose=True):
    '''
    Generates the distribution of network separation and interquartile range of network separation values during pruning for generating the different scores for deDANSy analysis. This is the key function of the algorithm that gathers all the results.

    Parameters:
    -----------
        - dedansy: deDANSy object
            The dansy object containing all expression and protein information
        - num_trials: int (Optional, Default 50)
            The number of trials for subsampling and random gene selection
        - pval_sweep: numpy array
            The threshold p-values to use for pruning analysis
        - return_distribution: bool
            Whether the distribution of p-values during the p-value sweep is return. Default is False as this is a niche request
        - processes: int
            Number of multiprocessing workers to use for analysis
        - verbose: bool
            Whether progress statements should be displayed
    
    Returns:
    --------
        - o: list of tuples
            The full distribution of values for each individual trial of the subsampled and randomly chosen genes
    '''
   
   # Setting up the inputs as necessary for the dependent functions by getting the information about the measured universe proteins
    orig_up = dedansy.up_DEGs
    orig_dn = dedansy.down_DEGs
    original_DEGs = list(orig_up) + list(orig_dn)
    deg_info = dedansy.retrieve_protein_info(original_DEGs)
    arch_lens = deg_info['Interpro Domain Architecture IDs'].apply(lambda x: len(x.split('|')))
    max_length = max([max(arch_lens),11])
    deg_arch_dist = np.histogram(arch_lens, bins=range(max_length+1))
    complete_arch_dist = dedansy.ref['Interpro Domain Architecture IDs'].apply(lambda x: len(x.split('|')))
    weight_list = deg_arch_dist[0][0:11]
    weight_list = np.append(weight_list,np.sum(deg_arch_dist[0][11:]))

    # Now normalizing the list 70% of the total number of DEGs available
    weight_list_n = np.round(weight_list/sum(weight_list)*(sum(weight_list)*0.7))
    up_frac = len(dedansy.up_DEGs)/len(deg_info)
    
    # For reproducibility (mostly only for when multiprocessing is enacted) creating a seed list that will be passed to the individual trials function
    seedlist = random.sample(range(50*num_trials), num_trials)
    if processes == 1:

        # Intializing some of the variables of the for loop
        rand_ns = np.zeros(num_trials)
        subsample_ns = np.zeros(num_trials)
        rand_iqr = np.zeros(num_trials)
        subsample_iqr = np.zeros(num_trials)
        
        if return_distributions: 
            rand_ns_dists = []
            subsample_ns_dists = []

        # Progress statement
        if verbose:
            print('Starting calculations')

        for i in range(num_trials):
            a = individual_trial_calc(dedansy, weight_list_n,complete_arch_dist,up_frac,pval_sweep,[orig_up,orig_dn],dist_flag=return_distributions, seed=seedlist[i])
            # Unpacking the values into the datastructures above
            rand_ns[i] = a[0]
            rand_iqr[i] = a[1]
            if return_distributions:
                rand_ns_dists.append(a[2])
                subsample_ns[i] = a[3]
                subsample_iqr[i] = a[4]
                subsample_ns_dists.append(a[5])
            else:
                subsample_ns[i] = a[2]
                subsample_iqr[i] = a[3]

    else:
        if verbose:
            print('Will do multiprocessing')

        # Setting up the arguments to be passed to multiple processes
        args = [(dedansy, weight_list_n,complete_arch_dist,up_frac,pval_sweep,[orig_up,orig_dn],return_distributions, seedlist[i]) for i in range(num_trials)]
        
        if __name__ == 'dansy.enrichment_helpers':
            pool = mp.Pool(processes=processes)
            with pool as p:
                a = p.starmap(individual_trial_calc, args,chunksize=5)
            pool.close() # In case but not necessary

        # Unpacking the values into the datastructures above
        rand_ns = [x[0] for x in a]
        rand_iqr = [x[1] for x in a]
        if return_distributions:
            rand_ns_dists = [x[2] for x in a]
            subsample_ns = [x[3] for x in a]
            subsample_iqr = [x[4] for x in a]
            subsample_ns_dists= [x[5] for x in a]
        else:
            subsample_ns = [x[2] for x in a]
            subsample_iqr = [x[3] for x in a]
    
    # Prepping the export
    if return_distributions:
        o = [rand_ns,rand_iqr,rand_ns_dists,subsample_ns,subsample_iqr,subsample_ns_dists]
    else:
        o = [rand_ns,rand_iqr,subsample_ns,subsample_iqr]

    return o

def retrieve_fpr_checks(dedansy,num_DEGs,fpr_trials = 50, num_internal_trials = 50, deg_ratios = 0.7, processes = 1, seed =123):
    '''
    Performs a false positive rate check on the values for the differentially expressed genes by performing the same subsampling and random gene selection analysis with random genes that were measured to determine if findings are true positives. This process can be highly time consuming based on the number of FPR trials that are to conducted.

    Parameters:
    -----------
        - dedansy: deDANSy object
            The dansy object containing all expression and protein information
        - num_DEGs: int
            The number of differentially expressed genes in the deDANSy object
        - fpr_trials: int (Optional, Default 50)
            The number of False Positive Rate trials to conduct
        - num_internal_trials: int (Optional, Default 50)
            The number of trials for subsampling and random gene selection that will be used for all the FPR trials. (Should match what was done on the analysis of interest)
        - deg_ratios: float
            Relative ratio of differentially expressed genes to designate as up or down
        - processes: int
            Number of multiprocessing workers to use for analysis
        - seed
            Random number generator seed

    Returns:
    --------
        - internal_fpr: list
            List of p-values for the network separation and interquartile range during pruning 
    '''
    
    #  Setting up the random DEGs generator
    rand_DEGs = dedansy.retrieve_random_ids(num=num_DEGs, iters=fpr_trials,seed = seed)

    internal_fpr = []
    
    for i in range(fpr_trials):
        cur_DEGs = next(rand_DEGs)
        cur_DEGs = sorted(cur_DEGs) # In case of weird issues with random sampling using sorted
        orig_up = random.sample(cur_DEGs, k=round(len(cur_DEGs)*deg_ratios))
        orig_dn = sorted(set(cur_DEGs).difference(orig_up))
        dedansy.set_DEG_ngrams(up_DEGs=orig_up, down_DEGs=orig_dn, verbose=False)
        temp = calculate_separation_stability(dedansy,num_trials=num_internal_trials,processes=processes,verbose = False)
        rand_iqr = [x for x in temp[1]]
        subsample_iqr = [x for x in temp[3]]
        rand_full_ns = [x for x in temp[0]]
        actual_full_ns = [x for x in temp[2]]

        # Now getting some of the stats and results
        temp_iqr_res = stats.mannwhitneyu(rand_iqr,subsample_iqr)
        temp_ns_res = stats.mannwhitneyu(rand_full_ns, actual_full_ns)

        internal_fpr.append((temp_ns_res[1], temp_iqr_res[1]))

    return internal_fpr

def calculate_fpr(actual_res, random_res):
    '''
    For each result of interest calculates the False positive rate

    Parameters:
    -----------
        - actual_res: list
            The results from conducting the analysis
        - random_res: list
            The p-values from the random FPR trials
    
    Returns:
    --------
        - tuple of FPR values
    
    '''
    ns_res = actual_res[0]
    iqr_res = actual_res[1]
    ns_rand_res = [x[0] for x in random_res]
    iqr_rand_res = [x[1] for x in random_res]

    # Now calculating the fpr values
    x = [i <= ns_res for i in ns_rand_res]
    ns_fpr = sum(x)/len(x)
    x = [i <= iqr_res for i in iqr_rand_res]
    iqr_fpr = sum(x)/len(x)

    return (ns_fpr, iqr_fpr)