import  pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as lines

from pybiomart import Dataset
from tqdm import tqdm
from enrichment_helpers import *


def get_max_info_enriched_ngrams(res_df, condition_labels = None, q = 0.05,):
    '''
    This plots the top X percent (default 5) n-grams enriched between two different conditions. For clarity n-grams that contain similar information will be collapsed into the shorter n-gram (i.e. if EGF-like domain and EGF-like domain|EGF-like domain both have similar enrichment values they will only be represented by EGF-like domain).
    
    Parameters:
    -----------
        res_df: pandas DataFrame
            A dataframe containing all the results including both the individual statistical enrichment and the False positive rate for all n-grams. (Note this will likely be removed once this is integrated into the actual module.)
        condition_labels: list (Optional)
            The labels for both conditions this should be provided in the order of up-regulated and down-regulated. (Note this will be removed once this is integrated into the actual module.)
        q: float (Optional)
            The quantile cutoff of values to plot
    
    Returns:
    --------
        seaborn/matplotlib plot

    '''

    p_thres = np.quantile(res_df['p'],q)
    filt_res_cands = res_df[res_df['p'] <= p_thres]['ngram']
    filt_res = res_df[res_df['ngram'].isin(filt_res_cands)].copy()
    if condition_labels != None:
        filt_res['variable'] = filt_res['variable'].map({'Up':condition_labels[0],'Down':condition_labels[1]})

    ngram_list = sorted(set(filt_res['ngram'].tolist()),key=lambda x:len(x.split('|')),reverse=True)
    ngrams_2_collapse = collapse_to_max_info(ngram_list,filt_res)    
    
    ngrams_kept = set(filt_res['ngram'].tolist()).difference(ngrams_2_collapse)
    maxinfo_filt_res = filt_res[filt_res['ngram'].isin(ngrams_kept)].copy()
    

    return maxinfo_filt_res

def plot_enriched_ngrams(res, dansyOI, condition_labels = None, q = 0.05,show_FPR=True ,**kwargs):
    max_res = get_max_info_enriched_ngrams(res, condition_labels, q)
    ngram_plot_names = {node:dansyOI.return_legible_ngram(node) for node in max_res['ngram'].tolist()}
    max_res['ngram'] = max_res['ngram'].map(ngram_plot_names)
    
    # Going through some of the default values that I have set up for the seaborn scatterplot and checking for them in the kwargs to overwrite my default values.
    sns_opts = {'palette':['deepskyblue','silver'],'edgecolor':'k','linewidth':0.5,'sizes':(1,40)}
    for opt in sns_opts:
        if opt in kwargs:
            sns_opts[opt] = kwargs[opt]
            del kwargs[opt] # Removing to ensure that seaborn does not error out.
        
    sns.scatterplot(max_res, x='variable',y='ngram',
                    size='-log10(p)', hue = 'FPR <= 0.05',
                    hue_order=[True, False],
                    **sns_opts,**kwargs)
    
    # If an axes is provided plot and adjust the legend to that specific axis
    if 'ax' in kwargs:
        handles, labels = kwargs['ax'].get_legend_handles_labels()  
        new_handles, new_labels = clean_legend(handles, labels,show_FPR)
        l = kwargs['ax'].legend(handles, labels, edgecolor='k', handletextpad=0.1, )
        
    else: 
        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles, new_labels = clean_legend(handles, labels,show_FPR)
        l = plt.legend(new_handles, new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1, )
    
    # Small aesthetic changes
    l.get_frame().set_linewidth(0.5)
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.25)
    plt.xlabel(None)
    plt.ylabel(None)

def plot_enriched_ngrams_presorted(res, dansyOI = None,x_order = 'geneset_order',ngram_ticks = None,show_FPR=True, **kwargs):
    max_res = res.copy()
   
    
    # Going through some of the default values that I have set up for the seaborn scatterplot and checking for them in the kwargs to overwrite my default values.
    sns_opts = {'palette':['deepskyblue','silver'],'edgecolor':'k','linewidth':0.5,'sizes':(1,40)}
    for opt in sns_opts:
        if opt in kwargs:
            sns_opts[opt] = kwargs[opt]
            del kwargs[opt] # Removing to ensure that seaborn does not error out.
    
    # Now getting some of the keyword arguments that are associated with the legend
    legend_opts = {'loc':'lower left', 'bbox_to_anchor':(1,1), 'handletextpad':0.1}
    for opt in legend_opts:
        if opt in kwargs:
            legend_opts[opt] = kwargs[opt]
            del kwargs[opt]
    
    sns.scatterplot(max_res, x=x_order,y='ngram_order',
                    size='-log10(p)', hue = 'FPR <= 0.05',
                    hue_order=[True, False],
                    **sns_opts,**kwargs)
    
    # Setting up the ticks assocaited with the ngrams if a dansy object and an n-gram order dict were provided.
    if ngram_ticks == None and dansyOI == None:
        pass
    elif ngram_ticks != None and dansyOI != None:
        ngram_plot_names = [dansyOI.return_legible_ngram(node) for node in ngram_ticks]
        plt.yticks(ticks=[v for v in ngram_ticks.values()], labels=ngram_plot_names)
        plt.ylim(max(list(ngram_ticks.values()))+0.5,-0.5)

    # If an axes is provided plot and adjust the legend to that specific axis
    if 'ax' in kwargs:
        handles, labels = kwargs['ax'].get_legend_handles_labels()  
        new_handles, new_labels = clean_legend(handles, labels,show_FPR)
        l = kwargs['ax'].legend(new_handles, new_labels, edgecolor='k', **legend_opts)
        
    else: 
        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles, new_labels = clean_legend(handles, labels,show_FPR)
        l = plt.legend(new_handles, new_labels, edgecolor='k',**legend_opts)
    
    # Small aesthetic changes
    l.get_frame().set_linewidth(0.5)

    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.25)
        
    plt.xlabel(None)
    plt.ylabel(None)

def clean_legend(handles, labels,show_FPR = True):

    if show_FPR:
        labels[0]= 'FPR$\leq$0.05'
        new_handles = handles
        new_labels = labels
        
    else:
        new_handles = [h for i,h in enumerate(handles) if i not in [0,1,2]]
        new_labels = [h for i,h in enumerate(labels) if i not in [0,1,2]]
    
    return new_handles, new_labels

def collapse_to_max_info(ngram_list, res_df):

    potential_collapse = {}
    for ngram in ngram_list:
        for inner_ngram in ngram_list:
            if inner_ngram != ngram and ngram in inner_ngram:
                if ngram not in potential_collapse:
                    potential_collapse[ngram] = []
                potential_collapse[ngram].append(inner_ngram)


    # Now for each of these checking the FPR and p-values to see if they should be collapsed
    ngrams_2_collapse = set()
    for ngram, children in potential_collapse.items():
        parent_p = res_df[res_df['ngram'] == ngram]['p'].tolist()
        parent_fpr = res_df[res_df['ngram'] == ngram]['FPR <= 0.05'].tolist()
        parent_cond = res_df[res_df['ngram'] == ngram]['variable'].tolist()
        if len(parent_p) == 1:
            for child in children:
                child_p = res_df[res_df['ngram'] == child]['p'].tolist()
                child_fpr = res_df[res_df['ngram'] == child]['FPR <= 0.05'].tolist()
                child_cond = res_df[res_df['ngram'] == child]['variable'].tolist()
                if len(child_p) == 1:
                    if child_cond == parent_cond:
                        if child_p < parent_p and parent_fpr != child_fpr:
                            pass 
                        elif parent_fpr != child_fpr:
                            pass 
                        else:
                            ngrams_2_collapse.add(child)
        else:
            for child in children:
            
                child_p = res_df[res_df['ngram'] == child]['p'].tolist()
                child_fpr = res_df[res_df['ngram'] == child]['FPR <= 0.05'].tolist()
                child_cond = res_df[res_df['ngram'] == child]['variable'].tolist()
                if len(child_cond) == 2:
                    # Checking both the p-vals
                    if any(c < p for c,p in zip(child_p,parent_p)) and any(p != c for c,p in zip(child_fpr,parent_fpr)):
                        pass
                        
                    elif any(p != c for c,p in zip(child_fpr,parent_fpr)):
                        pass
                    else:
                        ngrams_2_collapse.add(child)
    
    return ngrams_2_collapse

def gather_enrichment_results(hyper_values, fpr_values, alpha = 0.05):
    '''
    This gathers both the statistical results from the enrichment analysis and the FPR calculations to return a complete results dataframe.

    Parameters:
    -----------
        - hyper_values: dict
            Key-value pairs of both conditions that contains dictionaries with key-value pairs of n-grams and their statistical enrichent in each condition.
        - fpr_values: dict
            Key-value pairs of each n-gram for the different conditions in a comparison

    Returns:
    --------
        - res: pandas DataFrame
            Results dataframe that has aggregated all the n-gram statistical results
    '''
    fpr_df = pd.DataFrame().from_dict(fpr_values)
    hyper_df = pd.DataFrame().from_dict(hyper_values)
    hyper_df = hyper_df.melt(ignore_index=False, value_name='p')
    hyper_df['ngram'] = hyper_df.index
    fpr_df = fpr_df.melt(ignore_index=False, value_name='FPR')
    fpr_df['ngram'] = fpr_df.index
    res = hyper_df.merge(fpr_df)
    res.dropna(subset=['p'],inplace=True)
    res['-log10(p)'] = -np.log10(res['p'])
    res['FPR <= 0.05'] = res['FPR'] <= alpha
    return res

def calc_ngram_fpr_vals(hyper_vals, rand_ngram_pvals):
    '''
    Calculates the FPR for enriched n-grams in both conditions.

    Parameters:
    -----------
        hyper_vals: dict
            Dict of dict that contain the enrichment results of each n-gram for both conditions.
        rand_ngram_pvals: dict
            Dict of dicts that contains the equivalent enrichment results for n-grams from randomly chosen genes.
    
    Returns:
    --------
        fpr_dict: dict
            Dict of dict for each condition that contains the FPR values of each n-gram.
    '''
    # Now calculating the fpr for each of the nodes found within the actual network
    
    
    fpr_dict = {k:{} for k in hyper_vals}
    for c_dir,i in hyper_vals.items():
        for node in i:
            actual_p = i[node]
            if node in rand_ngram_pvals[c_dir]:
                rand_p_vals = rand_ngram_pvals[c_dir][node]
                num_fp = sum([x < actual_p for x in rand_p_vals])
                fpr_dict[c_dir][node] = num_fp/len(rand_p_vals)
            else:
                fpr_dict[c_dir][node] = 0

    return fpr_dict


# Defining helper functions to gather the results and plot them.
def get_plotting_data(res,metric):
    '''
    Gets the data that will be plotted in the bubble plot. Acceptable metrics are NS or IQR only.

    Parameters:
    -----------
        res: pandas DataFrame
            The imported data from running the enrichment analysis using the enrichment script.
        metric: str
            Which metric to retrieve.

    Returns:
    --------
        plot_data: pandas DataFrame
            The data that will be used for the plot
    '''
    if metric in['NS', 'IQR']:
        cat_map = {True:0, False:1}
    else:
        raise ValueError('Improper metric requested.')

    # Putting the data into the proper format
    plot_data = res[res['Metric'] == metric].drop("Metric", axis = 1)
    plot_data = plot_data.pivot(index = 'Comparison',columns = ['variable'])
    plot_data = plot_data['value'].reset_index()
    plot_data.columns.name = None

    # Adding in a few new variables related to the plot
    plot_data['FPR_Binary'] = plot_data[metric+'_FPR'] <= 0.05
    plot_data['Sign'] = plot_data[metric+' Cohen'] > 0
    plot_data['Score'] = np.abs(plot_data[metric+' Cohen'])
    plot_data['Category'] = plot_data['Sign'].map(cat_map)
    
    # Adding in some dummy variables to force some consistent sizes
    if metric == 'NS':
        for i in range(6):
            plot_data.loc[i+len(plot_data)] = None
            plot_data.loc[i+len(plot_data), 'Score'] = i
    else:
        # The IQR scores tend to be between 0-2 so getting the closest integer and then taking 5 steps to force the size
        max_vals = np.ceil(plot_data['Score'].max()) 
        for i,v in enumerate(np.linspace(0,max_vals,5)):
            plot_data.loc[i+len(plot_data)] = None
            plot_data.loc[i+len(plot_data), 'Score'] = v

    return plot_data


def plot_functional_scores(res, show_FPR_handle=True, aspect = 0.9):
     
    # Setting up the data and base plot
    data_plot = get_plotting_data(res, 'NS')
    _, axs = plt.subplots(1,2,figsize = (3,1.5))
    plt.subplot(1,2,1)
    sns.scatterplot(data_plot, y = 'Comparison', size='Score', x = 'Category',
    hue='FPR_Binary',sizes = (1,50), size_norm = (0,5),
                    palette=['mediumorchid', 'silver'],hue_order=[True, False],
                    linewidth = 0.5,edgecolor='k')
    plt.gca().set_aspect(aspect)

    # Adding in labels
    plt.title('Separation Relative\nto Random Genes',fontdict={'size':5})
    plt.xlabel(None)

    # Ensuring both ticks are present even if one of the categories is missing.
    plt.xticks([0,1],['More', 'Less'], rotation=45, ha='right')
    plt.xlim(-0.5,1.5)
    
    # Cleaning up the legend slightly
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = clean_up_legend(handles, labels, show_FPR_handle)
    l = plt.legend(new_handles,new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1)
    l.get_frame().set_linewidth(0.5)
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.5)
        
    # When there are several comparisons it is a little easier to digest when there are grid lines
    if len(set(data_plot['Comparison'].tolist())) >= 3:
        plt.grid(visible=True,axis='y', linewidth =0.25, linestyle= ':')

    # Now the IQR version
    data_plot = get_plotting_data(res, 'IQR')
    plt.subplot(1,2,2)
    sns.scatterplot(data_plot, y = 'Comparison', size='Score', x = 'Category',
    hue='FPR_Binary',sizes = (1,50),
                    palette=['seagreen', 'silver'],hue_order=[True, False],
                    linewidth = 0.5,edgecolor='k', legend='brief')
    
    plt.xticks([0,1],['Stably Distinct', 'Unstable/Overlap'], rotation=45, ha='right')
    plt.xlabel(None)
    plt.title('Enriched \nn-gram neighborhoods',fontdict={'size':5})
    plt.xlim([-0.5,1.5])
    plt.gca().set_aspect(aspect)
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = clean_up_legend(handles, labels, show_FPR_handle)
    l = plt.legend(new_handles,new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1)
    l.get_frame().set_linewidth(0.5)
    # Removing specific handles since they add more than necessary
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.5)
            
    plt.ylabel(None)
    plt.tick_params('y',labelleft=None)
    
    if len(set(data_plot['Comparison'].tolist())) >=3:
        plt.grid(visible=True,axis='y', linewidth =0.25, linestyle= ':')
        
    return axs

def clean_up_legend(handles, labels, show_FPR = True):
    
    if show_FPR:
            handles_2_rm = [5,7,9]
    else:
        handles_2_rm = [0,1,2,5,7,9] #Will not always need the FPR legend details so will remove them as well
    
    new_h = [h for i,h in enumerate(handles) if i not in handles_2_rm]
    new_l = [h for i,h in enumerate(labels) if i not in handles_2_rm]
    
    if show_FPR:
        new_l[0] = 'FPR$\leq$0.05'
        new_h.insert(3,new_h[3])
        new_l.insert(3,'')
    
    return new_h, new_l

def format_results(dataset_results):
    
    variables_2_keep = ['NS Cohen', 'IQR Cohen', 'NS p_log', 'IQR p_log', 'NS_FPR', 'IQR_FPR']
    res = dataset_results[dataset_results['variable'].isin(variables_2_keep)].copy()
    res['Metric'] = res['variable'].str.split(' ').str[0]
    res['Metric'] = res['Metric'].str.split('_').str[0]

    return res
    