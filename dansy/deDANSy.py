import warnings
import random
import itertools
import time
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import dansy.ngramUtilities as ngramUtilities
import dansy.dedansy_algorithm as algorithm
from dansy.dansy import dansy
from dansy.network_separation_helpers import network_separation,build_network_reference_dict
from dansy.enrichment_plotting_helpers import plot_functional_scores, gather_enrichment_results, calc_ngram_fpr_vals, plot_enriched_ngrams


class DEdansy(dansy):
    '''
    A container class of multiple Domain n-gram networks related to a differentially expressed dataset that was generated using the DESeq analysis pipeline. This provides methods to analyze and contrast pairs of domain architecture subnetworks to understand changes in functional molecular ecosystems available to different conditions.

    Parameters
    ----------
    dataset: pandas DataFrame
        The expression dataset that contains the proteins of interest and expression values to designate differentially expressed genes/proteins
    uniprot_ref: pandas DataFrame (Optional)
        The base reference file that contains all the proteins of interest within the dataset. Note: It is recommended to include if multiple instances of a DEdansy object are being instantiated that share a common set of proteins.
    n: int (Optional)
        Length of n-grams to extract. Default is 10
    id_conv: pandas DataFrame (Optional)
        A dataframe with a column of UniProt IDs and a column for a IDs used in the expression dataset to help in converting
    conv_cols: str (Optional)
        The name of the column with IDs matching in the id_conv dataframe. Assumes the naming convention generated from pybiomart
    data_ids: str (Optional)
        The name of the column in the dataset dataframe for the IDs to be converted to UniProt IDs
    penalty: 'dynamic' or int (Optional)
        The value or 'dynamic' for the penalty during network separation calculations. 
    run_conversion: bool
        Whether the dataset IDs have to be converted using a provided id_conv dataframe
    kwargs
        Additional keyword arguments for generating the DANSy network. See dansy.set_ngram_parameters for details acceptable values are reproduced below
            - 'min_arch'
            - 'max_node_len'
            - 'collapse'
            - 'readable_flag'
            - 'verbose'

    Attributes
    ----------
    -----------------
    At Initialization
    -----------------
    dataset: pandas DataFrame
        The expression dataset for DANSy analysis
    ref: pandas DataFrame
        The reference file information for the proteins within the dataset
    n: int
        The maximum length of n-grams being extracted
    interproIDs: list
        A list of all protein domain InterPro IDs that were found within the dataset
    protsOI: list
        The UniProt IDs for the proteins found within the dataset
    ngrams: list
        The extracted domain n-grams
    collapsed_ngrams: list
        The domain n-grams which were collapsed into other n-grams which represent the set of proteins
    G: networkx Graph
        The network graph representation of the DANSy n-gram network
    adj: pandas DataFrame
        The adjacency matrix for the n-gram network for the DANSy analysis
    interpro2uniprot: dict
        The keys of InterPro IDs with values of a list of UniProt IDs that have the InterPro ID
    id_conversion_dict: dict
        A dictionary containing the conversion of the provided gene/protein ID to a UniProt ID. If UniProt IDs were provided, returns a dict of UniProt to UniProt IDs.
    data_id_cols: str
        The ID column used in the dataset for conversion
    network_params: dict
        Key-value pairs of acceptable networkx drawing parameters
    min_arch: int (Default: 1)
        The minimum number of domain architectures for an n-gram to be retained.
    max_node_len: int
        The maximum n-gram length that will be retained during the collapsing step to represent n-grams sharing the same set of proteins. This will not be larger than n (Default of 10).
    collapse: bool
        Whether the n-grams were collapsed
    readable_flag: bool
        Whether the n-grams are human-legible
    verbose: bool
        Whether progress statements are to be printed during calculations
    
    --------------------------------------
    After Establishing Comparison MetaData 
    --------------------------------------
    comp_metadata: pandas DataFrame
        The metadata information for comparisons of interest in the dataset

    ----------------------
    After DEG Calculations
    ----------------------
    up_DEGs/down_DEGs: list
        List of UniProts that have were designated as up or down for a specific condition
    up_ngrams/down_ngrams: list
        The n-grams for either up- or down-regulated proteins/genes
    alpha: float
        The p-value cutoff for designating DEGs
    fcthres: float
        The fold-change cutoff for designating DEGs
    pval_data_col: str
        The column name in the datasetused for the p-value data for DEGs
    fc_data_col: str
        The column name in the dataset used for fold-change data for DEGs
    
    -----------------------------
    After creating deDANSy scores
    -----------------------------
    scores: pandas DataFrame
        The scores, p-values, and FPR values for the deDANSy Separation and Distinction scores for the comparisons
    raw_dists: pandas DataFrame
        The raw distribution of values for each score
    fpr_dists: pandas DataFrame
        The raw distribution of FPR values for each score
        


    '''
    def __init__(self,dataset,uniprot_ref = None,n = 10, id_conv = None, conv_cols = 'Gene stable ID', data_ids = 'gene_id',  penalty = 'dynamic',run_conversion = True, **kwargs):
        
        # Bare minimum attributes required for setting up an empty n-gram network.
        self.dataset = dataset
        self.ref = uniprot_ref
        self._n = n
        self.interproIDs = None
        
        # Converting the dataset ids for individual genes to the UniProt IDs
        if 'dbl_check' in kwargs:
            check_IDs_flag = kwargs['dbl_check']
        else:
            check_IDs_flag = False
        
        if run_conversion:
            if id_conv is None:
                raise ValueError('Missing an ID converting dataframe.')
            else:
                self.id_conversion_dict = create_id_conv(dataset, id_conv, conv_cols,data_ids,check_IDs_flag)
                self.protsOI = convert_2_uniprotIDs(dataset,id_conv, conv_cols,data_ids, check_IDs_flag)
        else:
            self.protsOI = dataset[data_ids].tolist()
            self.id_conversion_dict = {k:[k] for k in self.protsOI}
        # Saving the data_id column for instances when the DEGs have to be calculated.
        if isinstance(data_ids, list):
            self.data_id_col = data_ids[0]
        else:
            self.data_id_col = data_ids

        self.conversion_id_col = conv_cols

        # Now making sure there is a common reference that can be used for generating the n-gram networks.
        if self.ref is None:
            self.add_ref_df()
        
        self.populate_ngramNet(**kwargs)
        if self.verbose:
            print('Building the reference network information.')
        self.ref_data = build_network_reference_dict(self, penalty=penalty)


    def DEG_network_sep(self, force_run = False):
        ''' 
        This computes the network separation of two conditions that designates individual genes as differentially expressed. Silently passes a nan if it fails.
        '''

        # Get the UniProt IDs for the differentially expressed genes
        if hasattr(self, 'up_DEGs') and hasattr(self, 'down_DEGs'):
            pass
        else:
            raise ValueError('DEGs do not exist. Please run calc_DEG_ngrams.')
        
        if len(self.up_ngrams) == 0 or len(self.down_ngrams) == 0:
            ns = np.nan
        else:
            try:
                # Generating networks for the individual DEG conditions
                up_net = self.G.subgraph(self.up_ngrams)
                dn_net = self.G.subgraph(self.down_ngrams)
                ns = network_separation(up_net,dn_net, self.ref_data,force_run=force_run)
            except:
                ns = np.nan
        return ns
    
    def calc_DEG_ngrams(self, comp, batch_mode = False):
        '''
        Defines the DEG UniProt IDs and the associated n-grams for the datasset of interest
        '''
        # Checking if there was a set of DEGs that already exists and printing statement saying it will be overwritten.
        if hasattr(self,'up_DEGs'):
            verbose_flag = True
            if hasattr(self, 'G_collapsed'):
                del self.G_collapsed
        else:
            verbose_flag = False

        # This is only for running several and wanting to keep track of general progress.
        if batch_mode:
            verbose_flag = False

        # Now extracting the data columns which for the comparison of interest to define the differentially expressed genes/proteins
        compOI_info = self.comp_metadata.loc[comp]
        data_cols = [compOI_info['fc_col'], compOI_info['pval_col']]
        fc_thres = compOI_info['fc_thres']
        alpha = compOI_info['alpha']

        # Reducing large dataframe to what is the info needed for generating the DEG networks
        cols = [self.data_id_col]+data_cols
        dataset_OI = self.dataset.filter(cols)

        # Getting the differentially expressed genes
        deg_up = list(dataset_OI[(dataset_OI[data_cols[1]] <= alpha) & (dataset_OI[data_cols[0]] > fc_thres)][self.data_id_col])
        deg_dn = list(dataset_OI[(dataset_OI[data_cols[1]] <= alpha) & (dataset_OI[data_cols[0]] < -fc_thres)][self.data_id_col])

        # Keeping record of the data columns for a summary
        self.pval_data_col = data_cols[1]
        self.fc_data_col = data_cols[0]
        # Recording the thresholds for DEG values
        self.alpha = alpha
        self.fcthres = fc_thres
        # And converting to the UniProt IDs to highlight n-grams within the network that were retained for each
        up_DEGs = [v for k, v in self.id_conversion_dict.items() if k in deg_up]
        up_DEGs = set(list(itertools.chain.from_iterable(up_DEGs)))
        down_DEGs =  [v for k, v in self.id_conversion_dict.items() if k in deg_dn]
        down_DEGs = set(list(itertools.chain.from_iterable(down_DEGs)))

        # Now just due to random sampling issues making sure the DEGs are in alphabetical order and converting to a list
        up_DEGs = sorted(up_DEGs)
        down_DEGs = sorted(down_DEGs)

        self.set_DEG_ngrams(up_DEGs,down_DEGs, verbose=verbose_flag)

    def set_DEG_ngrams(self, up_DEGs, down_DEGs,collapse = True, verbose=True):
        '''
        This actually sets the DEGs so that they are calculated, but allows for custom sets to be generated for very specific purposes. It is not recommended to use this
        '''
        if hasattr(self,'up_DEGs') and verbose:
            print('Will be overwriting existing DEG data.')

        # Setting the DEGs
        self.up_DEGs = up_DEGs
        self.down_DEGs = down_DEGs

        # Now getting the n-grams associated with the collective DEGs 
        up_ngram_cands = [k for k,v in self.interpro2uniprot.items() if set(v).intersection(self.up_DEGs)]
        down_ngram_cands = [k for k,v in self.interpro2uniprot.items() if set(v).intersection(self.down_DEGs)]

        # Getting the n-gram candidates
        up_ngram_dict = {k:set(v).intersection(self.up_DEGs) for k,v in self.interpro2uniprot.items() if k in up_ngram_cands}
        down_ngram_dict = {k:set(v).intersection(self.down_DEGs) for k,v in self.interpro2uniprot.items() if k in down_ngram_cands}
        
        # Now collapsing these to the non-redundant ones
        if collapse:
            up_ngram_dict,_ = ngramUtilities.concatenate_ngrams(up_ngram_dict)
            down_ngram_dict,_ = ngramUtilities.concatenate_ngrams(down_ngram_dict)
            

        # Exporting to the object
        self.up_ngrams = [k for k in up_ngram_dict.keys()]
        self.down_ngrams = [k for k in down_ngram_dict.keys()]


    def plot_DEG_ns(self,pos = [],deg_labels=[], large_cc_mode=False):
        '''
        Using the defined differentially expressed genes displaying a network graph of the n-grams that are associated or shared between the DEG conditions.
        '''

        # Setting default labels otherwise setting the label names for the legend.
        if deg_labels:
            up_label = deg_labels[0]
            down_label = deg_labels[1]
        else:
            up_label = 'Up'
            down_label = 'Down'

        # Retrieving the n-grams across both DEG conditions to filter out any isolates/connected components in the reference network that are not used.
        all_deg_ngrams = set(self.up_ngrams).union(self.down_ngrams)

        if large_cc_mode:
            large_ref_cc = max(nx.connected_components(self.G), key=len)
            all_deg_ngrams = all_deg_ngrams.intersection(large_ref_cc)
        
        # Setting up the node color list for plotting
        node_colors = []
        for node in self.G.subgraph(all_deg_ngrams):
            if (node in self.up_ngrams) and (node in self.down_ngrams):
                node_colors.append('tab:purple')
            elif node in self.up_ngrams:
                node_colors.append('tab:cyan')
            elif node in self.down_ngrams:
                node_colors.append('tab:red')
            else:
                node_colors.append('tab:gray')
        
        # Dropping connected components not shared with the DEGs and the reference network
        if large_cc_mode:
            cc_2_keep = set(large_ref_cc)
        else:
            cc_2_keep = set()
            for cc in nx.connected_components(self.G):
                if set(cc).intersection(all_deg_ngrams):
                    cc_2_keep.update(cc)

        # Getting the node positions for network drawing if not supplied.
        if pos == [] and hasattr(self, 'network_params'):
            if 'pos' in self.network_params:
                pos = self.network_params['pos']
            else:
                pos = nx.spring_layout(self.G, k=0.05)
        elif pos ==[]:
            pos = nx.spring_layout(self.G, k=0.05)
        
        # Getting all the basic network parameters that are default in the n-gram networks
        # Default values 
        basic_network_params = {'node_size':1,
                                  'edgecolors':'k',
                                  'linewidths':0.1,
                                  'width':0.25,
                                  'edge_color':'#808080',
                                  }
        net_draw_params = {}
        for param in basic_network_params:
            if param in self.network_params:
                net_draw_params[param] = self.network_params[param]
            else:
                self.network_params[param] = basic_network_params[param]
                net_draw_params[param] = basic_network_params[param]

        # Now drawing and adding legends
        plt.figure(figsize=(2,2), dpi=300)
        nx.draw(self.G.subgraph(cc_2_keep), pos, node_color = 'tab:gray', alpha = 0.1, **net_draw_params)
        nx.draw(self.G.subgraph(all_deg_ngrams), pos, node_color = node_colors,**net_draw_params)

        # Legend drawing
        plt.scatter([],[],c='tab:cyan',s=1, label=up_label)
        plt.scatter([],[],c='tab:red',s=1, label=down_label)
        plt.scatter([],[],c='tab:purple',s=1, label='Both')
        plt.legend(bbox_to_anchor=(1,0.5),frameon=False)


    def deg_summary(self, detailed = False):
        '''
        This provides a summary of the DEG information that has been used within this dataset.
        '''

        summary_df = pd.DataFrame(index=['p-value threshold', 'Fold change threshold', 'Up Regulated DEGs','Down Regulated DEGs'],columns=[''])

        vals = [self.alpha, self.fcthres, len(self.up_DEGs), len(self.down_DEGs)]

        for i,v in zip(summary_df.index, vals):
            summary_df.loc[i] = v
        
        if detailed:
            extended_inds = ['Data ID column','p-val data column', 'FC data column','Up-ngrams','Down ngrams','Common n-grams']

            # Getting common ngram counts
            common = set(self.up_ngrams).intersection(self.down_ngrams)

            extended_vals = [self.data_id_col, self.pval_data_col,self.fc_data_col, len(self.up_ngrams), len(self.down_ngrams), len(common)]

            for i,v in zip(extended_inds, extended_vals):
                summary_df.loc[i] = v

        return summary_df
    
    def ngram_DEG_hypergeom(self, condition):
        '''
        Performs the hypergeometric over-representation test on the n-grams associated with different conditions.

        Parameters:
        -----------
            - condition: str
                Which condition to perform the test on must be either Up or Down (case insensitive)

        Returns:
        --------
            - p_vals: dict
                The p-value of each n-gram given the condition relative to the whole potential universe of the n-gram reference background of the n-gram network.
        '''

        # Make the condition input all lower case
        condition = condition.lower()

        if condition == 'up':
            ngramsOI = self.up_ngrams
            degsOI = self.up_DEGs
        elif condition == 'down':
            ngramsOI = self.down_ngrams
            degsOI = self.down_DEGs
        else:
            raise ValueError('Please specify either up or down.')
        
        p = ngram_enrichment(self, prots = degsOI, ngrams=ngramsOI)

        return p

    def create_contrast_metadata(self,comparisons, fc_cols, pval_cols,  fcs = 1,alphas = 0.05, delim=None):
        '''
        This create the metadata for the deDANSy object that contains information for each comparison of interest and the cutoffs to define differentially expressed genes/proteins (DEGs). Must provide the same number of columns for both the fold change and p-value columns

        Parameters
        -----------
        comparisons : list
            All comparisons that will be used that match what is provided in the deDANSy dataset
        fc_cols : str or list
            Either a stem or list of columns used for the foldchanges to define DEGs (Use the delim parameter and a stem if reoccurring patterns are used where the comparison is at the end.)
        pval_cols : str or list
            Either a stem or list of columns used for the p-values to define DEGs. (Use the delim parameter and a stem if reoccurring patterns are used where the comparison is at the end.)
        fcs : float or list
            The fold-change cutoff. Can either be a single value or a list of values of the same size as comparisons
        alphas : float or list
            The p-value cutoff. Can either be a single value or a list of values of the same size as comparisons
        delim : str (Optional)
            Delimiter used in column names separating the stem of the column from the comparison. If none then will take the 
    
        Returns
        --------
        comp_meta : pandas DataFrame
            DataFrame where each row is a single comparison
        
        '''

        # Checking parameter inputs
        fc_list_flag = False
        if isinstance(fcs, list):
            fc_list_flag = True
            if len(fcs) == 1:
                fcs = fcs[0]
                fc_list_flag = False
            if len(fcs) != len(comparisons):
                raise ValueError('Provide either a single value or list of values for each comparison of fold-change cutoffs')
        
        alpha_list_flag = False
        if isinstance(alphas, list):
            alpha_list_flag = True
            if len(alphas) == 1:
                alpha_list_flag = False
                alphas = alphas[0]
            if len(alphas) != len(comparisons):
                raise ValueError('Provide either a single value or list of values for each comparison of p-value cutoffs')
        
        col_lists = isinstance(fc_cols, list) and isinstance(pval_cols, list)

        if col_lists and len(fc_cols) != len(pval_cols):
            raise ValueError('The number of fold change columns and p-value columns does not match.')
        elif col_lists and len(fc_cols) != len(comparisons):
            if len(fc_cols) > 1:
                raise ValueError('The number of data columns does not match the number of comparisons.')
        
        col_strs = isinstance(fc_cols, str) and isinstance(pval_cols, str)
        if delim is None and col_strs:
            if len(comparisons) > 1:
                raise ValueError('Cannot create feasible data columns without a delimiter')

        if (not col_lists) and (not col_strs):
            raise ValueError('The data columns have to be either lists or strings not a mixture of the two.')

        # Now creating the DataFrame to populate with values
        comp_meta = pd.DataFrame(columns=['fc_col', 'fc_thres','pval_col','alpha'], index=comparisons)
        
        for i,c in enumerate(comparisons):
            if col_strs:
                if delim is None:
                    fcol = fc_cols
                    pcol = pval_cols
                else:
                    fcol = delim.join([fc_cols,c])
                    pcol = delim.join([pval_cols,c])
            else:
                fcol = fc_cols[i]
                pcol = pval_cols[i]

            comp_meta.loc[c,'fc_col'] = fcol
            comp_meta.loc[c,'pval_col'] = pcol
            
            # Get the corresponding values for the cutoffs
            if fc_list_flag:
                f = fcs[i]
            else:
                f = fcs

            if alpha_list_flag:
                a = alphas[i]
            else:
                a = alphas
            comp_meta.loc[c,'fc_thres'] = f
            comp_meta.loc[c,'alpha'] = a

        if hasattr(self, 'comp_metadata'):
            temp = pd.concat([self.comp_metadata, comp_meta], axis = 0)
            self.comp_metadata = temp
        else:
            self.comp_metadata = comp_meta

    def calculate_scores(self, comps, fpr_trials=50, min_pval = -10, num_ss_trials = 100, processes = 1, seed=None, verbose=True, overwrite=False):
        '''
        This calculates the separation and distinct functional neighborhood scores for a deDANSy instance. It creates new attributes for the deDANSy instance containing all the information of interest. If scores for a condition have been generated, this will raise a warning and overwrite existing scores.
        
        Parameters
        -----------
        dedansy : deDANSy object
            The base deDANSy object containing all n-grams and expression data
        comps : list or str
            The conditions that will be compared
        fpr_trials : int (Optional)
            Number of FPR trials to perform.
        min_pval : int (Optional)
            The log10 transform of the minimum p-value to use for the p-value pruning sweep step.
        num_ss_trials : int (Optional)
            The number of subsampled (and random) networks used to build distributions for comparing the network separation and IQR
        processes : int (Optional)
            Number of processes to use if multiprocessing is desired. (Recommended having 4-8 when feasible)
        seed : int
            Seed for random numbers. If not provided will use system time

        Returns
        --------
        The following attributes are added or adjusted:
        scores : pandas DataFrame
            All scores, p-values, and FPR values for each condition
        raw_dists : pandas DataFrame
            For each condition the raw values that made up the distributions for each score
        fpr_dists : pandas DataFrame
                For each condition the p-values from the random FPR trials for calculating the FPR'''
        
        # Check for existing scores for all conditions provided
        if hasattr(self, 'scores'):
            a = set(comps).intersection(self.scores.index)
            if len(a) >= 1:
                if overwrite:
                    warnings.warn('At least one condition has existing scores that will be overwritten.')
                else:
                    warnings.warn('At least one condition has existing scores that will be kept.')
        else:
            a = set()

        temp_scores, temp_raw_dists, temp_fpr_dists = algorithm.calculate_scores(self, comps, fpr_trials,min_pval, num_ss_trials,processes,seed, verbose)

        if hasattr(self, 'scores'):
            cur_scores = self.scores
            if overwrite:
                cur_scores.update(temp_scores)
            new_scores = cur_scores.combine_first(temp_scores)
            new_raw_dist = self.raw_dists.combine_first(temp_raw_dists)
            new_fpr_dist = self.fpr_dists.combine_first(temp_fpr_dists)
        else:
            new_scores = temp_scores
            new_raw_dist = temp_raw_dists
            new_fpr_dist = temp_fpr_dists
        
        self.scores = new_scores
        self.raw_dists = new_raw_dist
        self.fpr_dists = new_fpr_dist

    def plot_scores(self, show_FPR = True, aspect = 0.9, order = None):
        '''
        This creates the bubble plots for the separation and distinction scores. This is a wrapper function for the base plotting function found in the enrichment_plotting_helpers.

        Parameters
        ----------
        show_FPR : bool (Optional)
            Whether the FPR legend handles should be displayed.
        aspect : float (Optional)
            The aspect ratio for each score plot.
        order : dict (Optional)
            Key-value pairs for each comparison and what order they should be displayed on the axis.
        
        Returns
        --------
        ax : matplotlib Axes
            The axes of the resulting plot 

        '''
        x = self.scores.copy()
        x['Separation_Significance'] = x['Separation_FPR'] <= 0.05
        x['Distinction_Significance'] = x['Distinction_FPR'] <= 0.05
        ax = plot_functional_scores(x, show_FPR,aspect=aspect, order=order)
        
        return ax
    
    def calculate_ngram_enrichment(self,comparison, fpr_trials = 100, seed = None):
        '''
        Calculates the n-gram enrichment for the comparison of interest to find the most significant n-grams.

        Parameters:
        -----------
            - comparison: str
                The comparison of interest that contains the up- and down-regulated genes/proteins
            - alpha: float (Optional, Default: 0.05)
                Threshold of values to return
            - fpr_trials: int (Optional, Default: 100)
                The number of trials used to calculate the false positive rate
            - seed: int (Optional)
                Seed for the random state
        
        Return:
        -------
            Creates the new attribute
            - ngram_results: dict of pandas DataFrame
                All the n-gram enrichment values that pass the significance threshold for the comparison of interest
        '''
        # Setting the random state
        if seed is None:
            random.seed(int(time.time()))
        else:
            random.seed(seed)

        # First determine and store the enriched n-grams for the comparison of interest
        self.calc_DEG_ngrams(comparison)        
        enriched_ngrams = {}
        for i in ['Up', 'Down']:
            enriched_ngrams[i] = self.ngram_DEG_hypergeom(i)
        
        # Get the original DEGs for defining number of proteins to randomly choose
        true_up = self.up_DEGs
        true_dn = self.down_DEGs
        total_degs = len(set(true_up).union(true_dn))
        frac_up = len(true_up)/total_degs

        # Getting the random protein generator
        rand_genes = self.retrieve_random_ids(num=total_degs, iters=fpr_trials)
        rand_ngram_pvals = {'Up':{}, 'Down':{}}
        for i in range(fpr_trials):
            cur_DEGs = next(rand_genes)
            orig_up = random.sample(cur_DEGs, k=round(len(cur_DEGs)*frac_up))
            orig_dn = list(set(cur_DEGs).difference(orig_up))
            self.set_DEG_ngrams(up_DEGs=orig_up, down_DEGs=orig_dn, verbose=False)
            rand_up_hyper = self.ngram_DEG_hypergeom('Up')
            rand_dn_hyper = self.ngram_DEG_hypergeom('Down')
            for j,c_dir in zip([rand_up_hyper,rand_dn_hyper],['Up','Down']):
                for node in j:
                    if node not in rand_ngram_pvals[c_dir]:
                        rand_ngram_pvals[c_dir][node] = []
                    rand_ngram_pvals[c_dir][node].append(j[node])
        
        # Now getting the FPR values and putting the results in the correct order
        fpr_dict = calc_ngram_fpr_vals(enriched_ngrams, rand_ngram_pvals)
        res = gather_enrichment_results({'Up':enriched_ngrams['Up'],'Down':enriched_ngrams['Down']}, fpr_dict)
    
        if hasattr(self,'ngram_results'):
            self.ngram_results[comparison] = res
        else:
            self.ngram_results = {}
            self.ngram_results[comparison] = res

    def plot_ngram_enrichment(self, comparison, p = 0.05, q=0.05, show_FPR = True, **kwargs):
        '''
        Creates a bubble plot of the enriched n-grams for up and down-regulated n-grams for the comparison of interest. The resulting plot can be paritally customized based on additional keyword parameters. This function is a wrapper for the plot_enriched_ngrams function from the enrichment_plotting_helpers module, but specific to the deDANSy object instance.

        Parameters:
        -----------
            - comparison: str
                The comparison to plot the enriched n-grams of
            - q: float (Optional)
                The quantile cutoff of values to plot. Default (0.05)
            - p: float (Optional)
                The p-value threshold to limit n-grams to plot. Default is 0.05 but if combined with the quantile (q) can be lower.
            show_FPR: bool
                Flag whether to show the FPR legend portion.
            kwargs: optional keywords
                - 'palette','edgecolor', 'linewidth', 'sizes' that adjust the seaborn scatterplot aesthetics.
                - 'loc', 'bbox_to_anchor', 'handletextpad' to adjust matplotlib legend information.
                - 'ax' to specify a matplotlib Axes to plot onto
        
        Returns:
        --------
            seaborn/matplotlib plot
        '''

        resOI = self.ngram_results[comparison]
        plot_enriched_ngrams(resOI, dansyOI=self, p = p, q=q, show_FPR=show_FPR, **kwargs)
        plt.xlim(-0.5,1.5)
    
    def get_ngram_results(self,comparison):
        '''
        Returns the n-gram enrichment results of the comparison of interest.

        Parameters:
        -----------
            - comparison: str
                The comparison of interest
        
        Returns:
        --------
            - res: pandas DataFrame
                The n-gram enrichment dataframe
        '''

        temp = self.ngram_results[comparison]
        
        # Convert the n-gram IDs to legible names
        temp['ngram_legible'] = temp['ngram'].apply(lambda x: self.return_legible_ngram(x))

        # Now putting the columns into a more logical order
        col_order = ['variable', 'ngram_legible', 'ngram', 'p', '-log10(p)', 'FPR', 'FPR <= 0.05']
        temp = temp[col_order]

        return temp

## Helper functions for converting the IDs around
def convert_2_uniprotIDs(df, id_conv, conv_col = 'Gene stable ID', data_id_cols = 'gene_id', dbl_check = False):
    '''
    This takes a dataframe of interest and converts the specified column to UniProt IDs given a second dataframe consisting of UniProt IDs, ENSEMBL ids, and gene names/synonyms retrieved using biopython. If desired there will be a double check to ensure all IDs are retrieved if an archived version of ENSEMBL has been used and gene names are requested instead.
    '''

    # Now getting all possible UniProt IDs given the IDs of interest.
    ensembl_uni_dict = create_id_conv(df, id_conv=id_conv,conv_col=conv_col, data_id_cols=data_id_cols, dbl_check=dbl_check)
    uniprot_IDs = set(itertools.chain.from_iterable(ensembl_uni_dict.values()))

    return uniprot_IDs

def create_id_conv(df, id_conv, conv_col = 'Gene stable ID', data_id_cols = 'gene_id', dbl_check = False):
    '''
    This is an internal function that makes a dict to convert the ensembl IDs (by default) to UniProt IDs. 
    '''
    # Check if a list is provided for the data_id_cols or if a single/default value provided.
    if isinstance(data_id_cols, list):
        if dbl_check:
            data_id_col = data_id_cols[0]
        else:
            data_id_col = data_id_cols[0]
            warnings.warn('More than one column name was provided for converting, but a double check was not designated. Ignoring additional inputs.')
    else:
        data_id_col = data_id_cols

    # Retrieve all ENSEMBL IDs which are found within the dataset of interest
    success_mapping_ids = set(id_conv[id_conv[conv_col].isin(df[data_id_col])]['Gene stable ID'])
    
    if dbl_check:
        gene_dbl_chck = df[~df[data_id_col].isin(success_mapping_ids)][data_id_cols[1]].dropna()
        cands = id_conv[id_conv['Gene name'].isin(gene_dbl_chck)]
        success_mapping_ids.update(cands['Gene stable ID'])

    id_dict = id_conv[id_conv['Gene stable ID'].isin(success_mapping_ids)].filter(['Gene stable ID', 'UniProtKB/Swiss-Prot ID']).drop_duplicates().dropna().to_dict('tight')
    id_dict = id_dict['data']

    # Now to ensure that all potential UniProt IDs are accounted for have to go through and put them in lists as some Ensembl IDs map to multiple proteins.
    id_conversion = {}
    for conversion_data in id_dict:
        ensembl = conversion_data[0]
        uniprot = conversion_data[1]
        if ensembl in id_conversion:
            id_conversion[ensembl].append(uniprot)
        else:
            id_conversion[ensembl] = [uniprot]

    return id_conversion

# Helper functions for n-gram enrichment.

def ngram_enrichment(dansy, prots, collapse = True, **kwargs):
    '''
    Calculates the enrichment of n-grams associated with a subset of proteins within a DANSy object by Fisher's exact test.
    
    Note: This method can specifically analyze specific n-grams within the protein subset. However, this process should only be accessed with the deDANSy object and is not recommended to be used.

    Parameters
    ----------
        - dansy: DANSY
            The DANSy object which contains the proteins of interest. (This can be either the differential expression or standard class.)
        - prots: list
            List containing the proteins of interest for enrichment analysis.
        - collapse: bool
            Whether the n-grams should be collapsed to their most informative and non-redundant n-grams.
        - kwargs: optional keyword arguments (Not recommended to use)
            - ngrams: list
                List of n-grams that are to be analyzed specifically.
    '''

    p_vals = ngram_subset_enrichment(prots, dansy.protsOI, dansy, collapse=collapse, kwargs=kwargs)

    return p_vals

def ngram_subset_enrichment(protsOI, full_prots,dansy_bkg, collapse = True, **kwargs):
    '''Peform Fisher's exact test for n-grams found in a subset of proteins that may be found in the full list of proteins. This is the more general version that does not require using the full DANSy protein list.
    
        Parameters:
        -----------
            - protsOI: list
                List containing the proteins of interest for enrichment analysis.
            - full_prots: list
                List containing the proteins that make up the full background list.
            - dansy_bkg: DANSy object
                The DANSy that the proteins and n-grams are associated with
            - collapse: bool
                Whether the n-grams should be collapsed to their most informative and non-redundant n-grams.
            - kwargs: key, value mappings (Not recommended)
                Additional keyword arguments:
                    - ngrams: list
                        List of n-grams that are to be analyzed specifically.
            
        Returns:
        --------
            - p_vals: dict
                Key-value pairs of n-grams and their enrichment p-value.
    '''

    if 'ngrams' in kwargs:
        ngrams = kwargs['ngrams']
        
        # Making sure all n-grams are found in at least one of the proteins. If any are not raise an error.
        internal_check = [len(set(v).intersection(protsOI)) == 0 for k,v in dansy_bkg.interpro2uniprot.items() if k in ngrams]
        if any(internal_check):
            raise ValueError('At least one provided n-gram is not found in the proteins of interest.')
    else:
        ngrams = []

    subset_prots = list(set(protsOI).intersection(full_prots)) # Ensure that the foreground proteins are found within the background (removing any which are not) 
    M = len(subset_prots)
    N = len(full_prots)
    p_vals = {}
    
    if full_prots == dansy_bkg.protsOI:
        full_check = True
    else:
        full_check = False

    # Get the n-grams of the proteins of interest if they were not provided
    if ngrams:
        ngramsOI = ngrams
    else:
        ngram_cands = [k for k,v in dansy_bkg.interpro2uniprot.items() if set(v).intersection(subset_prots)]
        ngram_dict = {k:set(v).intersection(subset_prots) for k,v in dansy_bkg.interpro2uniprot.items() if k in ngram_cands}

        if collapse:
            ngram_dict,_ = ngramUtilities.concatenate_ngrams(ngram_dict)
        ngramsOI = [k for k in ngram_dict.keys()]
    
    # Now using the cdf/sf of the hypergeometric distribution to get p-values (This is equivalent to Fisher's exact)
    for node in ngramsOI:
        
        # Skip the filtering if using the full background
        if full_check:
            full_prot_list = dansy_bkg.interpro2uniprot[node]
        else:
            
            full_prot_list = [u for u in dansy_bkg.interpro2uniprot[node] if u in full_prots]
            
        # Now getting the numbers for the hypergeom distribution and calculating the cdf (or really sf since it is equivalent to 1-cdf)
        k = len(set(full_prot_list).intersection(subset_prots))
        n = len(set(full_prot_list))
        p = stats.hypergeom.sf(k-1,N, n,M)
        p_vals[node] = p

    return p_vals