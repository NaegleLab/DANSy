import  pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as lines
from dansy.enrichment_helpers import *


def get_max_info_enriched_ngrams(res_df, condition_labels = None, q = None,p = None):
    '''
    Returns the top X values of enriched n-grams that passes a quantile and/or p-value cutoff. This will collapse n-grams that provide similar p-value trends into a single representative n-gram if a shorter n-gram is in the longer n-gram. Longer n-grams that have more signficant p-values than their shorter counterparts will be retained.

    Parameters:
    -----------
        - res_df: pandas DataFrame
            Dataframe of all the n-gram enrichment p-values for each condition
        - condition_labels: list (Optional)
            list of strings that are labels for the different conditions. If not provided then defaults to Up and Down
        - q: float (Optional)
            quantile cutoff for p-values to return n-grams. Default value is 0.05.
        - p: float (Optional)
            p-value cutoff for n-grams to return. If provided with q this will set the upper bound of p-values.

    Returns:
    --------
        - maxinfo_filt_res: pandas DataFrame
            A filtered version of the res_df provided to only n-grams that pass the cutoffs provided
    
    '''

    # Checking the p-value and quantile cutoffs to determine a threshold.
    if p is None and q is None:
        q = 0.05
        p_thres = np.quantile(res_df['p'],q)
    elif p is not None and q is not None:
        x = np.quantile(res_df['p'],q)
        p_thres = np.min([p, x])
    elif p is not None:
        p_thres = p
    else:
        p_thres = np.quantile(res_df['p'],q)

    # Initial filtering based on only the p-value threshold
    filt_res_cands = res_df[res_df['p'] <= p_thres]['ngram']
    filt_res = res_df[res_df['ngram'].isin(filt_res_cands)].copy()

    # Creating the condition labels
    if condition_labels != None:
        filt_res['variable'] = filt_res['variable'].map({'Up':condition_labels[0],'Down':condition_labels[1]})

    # Now sorting the n-grams based on their length to start the collapsing step
    ngram_list = sorted(set(filt_res['ngram'].tolist()),key=lambda x:len(x.split('|')),reverse=True)

    # Collapsing the n-grams based on their p-values and if they have similar trends or not.
    ngrams_2_collapse = collapse_to_max_info(ngram_list,filt_res)    
    ngrams_kept = set(filt_res['ngram'].tolist()).difference(ngrams_2_collapse)

    # Filtering the results dataframe to only the collapsed n-grams
    maxinfo_filt_res = filt_res[filt_res['ngram'].isin(ngrams_kept)].copy()

    return maxinfo_filt_res

def plot_enriched_ngrams(res, dansyOI, condition_labels = None, q = 0.05,p = None, show_FPR=True ,**kwargs):
    '''
    This plots the top X percent (default 5) n-grams enriched between two different conditions. For clarity n-grams that contain similar information will be collapsed into the shorter n-gram (i.e. if EGF-like domain and EGF-like domain|EGF-like domain both have similar enrichment values they will only be represented by EGF-like domain).
    
    Parameters:
    -----------
        res_df: pandas DataFrame
            A dataframe containing all the results including both the individual statistical enrichment and the False positive rate for all n-grams. (Note this will likely be removed once this is integrated into the actual module.)
        dansyOI: deDANSy object
            The deDANSy object that contains the n-grams of interest
        condition_labels: list (Optional)
            The labels for both conditions this should be provided in the order of up-regulated and down-regulated. (Note this will be removed once this is integrated into the actual module.)
        q: float (Optional)
            The quantile cutoff of values to plot. Default (0.05)
        p: float (Optional)
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
    max_res = get_max_info_enriched_ngrams(res, condition_labels, q,p)
    ngram_plot_names = {node:dansyOI.return_legible_ngram(node) for node in max_res['ngram'].tolist()}
    max_res['ngram'] = max_res['ngram'].map(ngram_plot_names)
    
    # Going through some of the default values set up for the seaborn scatterplot and checking for them in the kwargs to overwrite default values.
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

    sns.scatterplot(max_res, x='variable',y='ngram',
                    size='-log10(p)', hue = 'FPR <= 0.05',
                    hue_order=[True, False],
                    **sns_opts,**kwargs)
    
    # If an axes is provided plot and adjust the legend to that specific axis
    if 'ax' in kwargs:
        handles, labels = kwargs['ax'].get_legend_handles_labels()  
        new_handles, new_labels = clean_ngram_legend(handles, labels,show_FPR)
        l = kwargs['ax'].legend(handles, labels, edgecolor='k', handletextpad=0.1, )
        
    else: 
        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles, new_labels = clean_ngram_legend(handles, labels,show_FPR)
        l = plt.legend(new_handles, new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1, )
    
    # Small aesthetic changes
    l.get_frame().set_linewidth(0.5)
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.25)
    plt.xlabel(None)
    plt.ylabel(None)

def plot_enriched_ngrams_presorted(res,x_order = 'variable', dansyOI = None,ngram_ticks = None,show_FPR=True, **kwargs):
    '''
    This plots a presorted n-gram enrichment dataframe based on a provided n-gram order and provided order for conditions (i.e. x-axis).
    
    Parameters:
    -----------
        res: pandas DataFrame
            A dataframe containing all the results including both the individual statistical enrichment and the False positive rate for all n-grams that has been presorted.
        x_order: str
            The column name to use for plotting on the x-axis. Default is variable assuming it was from a prior n-gram enrichment that was sorted in a different method.
        dansyOI: deDANSy object
            The deDANSy object that contains the n-grams of interest
        ngram_ticks: dict
            Key-value pairs of the n-grams and their order
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
    
    # Setting up the ticks associated with the ngrams if a dansy object and an n-gram order dict were provided.
    if ngram_ticks == None and dansyOI == None:
        pass
    elif ngram_ticks != None and dansyOI != None:
        ngram_plot_names = [dansyOI.return_legible_ngram(node) for node in ngram_ticks]
        plt.yticks(ticks=[v for v in ngram_ticks.values()], labels=ngram_plot_names)
        plt.ylim(max(list(ngram_ticks.values()))+0.5,-0.5)

    # If an axes is provided plot and adjust the legend to that specific axis
    if 'ax' in kwargs:
        handles, labels = kwargs['ax'].get_legend_handles_labels()  
        new_handles, new_labels = clean_ngram_legend(handles, labels,show_FPR)
        l = kwargs['ax'].legend(new_handles, new_labels, edgecolor='k', **legend_opts)
        
    else: 
        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles, new_labels = clean_ngram_legend(handles, labels,show_FPR)
        l = plt.legend(new_handles, new_labels, edgecolor='k',**legend_opts)
    
    # Small aesthetic changes
    l.get_frame().set_linewidth(0.5)

    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.25)
        
    plt.xlabel(None)
    plt.ylabel(None)

def clean_ngram_legend(handles, labels,show_FPR = True):
    '''
    Cleans up the n-gram enrichment legend to show only relevant information if the FPR is to be displayed or not.

    Parameters:
    -----------
        - handles: list
            List of matplotlib legend handles to adjust
        - labels: list
            List of maptlotlib legend labels to adjust
        - show_FPR: bool
            Whether the FPR portion of the legend should be displayed
    
    Returns:
    --------
        - new_handles: list
            The new handles to input into the legend
        - new_labels: list
            The new labels to input into the legend
    '''

    labels[1]= 'FPR$\leq$0.05'
    labels[2] = 'FPR > 0.05'
    if show_FPR:
        # Dropping the FPR legend title since it is provided in the actual labels
        new_handles = handles[1:len(handles)]
        new_labels = labels[1:len(labels)]
        
    else:
        new_handles = [h for i,h in enumerate(handles) if i not in [0,1,2]]
        new_labels = [h for i,h in enumerate(labels) if i not in [0,1,2]]
    
    return new_handles, new_labels

def collapse_to_max_info(ngram_list, res_df):
    '''
    This collapses the n-grams to those that represent the most discriminating information of interest. This will take longer n-grams and collapse them into shorter ones if the trends of p-values are similar, but the longer n-grams are slightly less signficant. If a longer n-gram is more significant it will not be collapsed.

    Parameters:
    -----------
        - ngram_list: list
            List of n-grams to consider for collapsing
        - res_df: pandas DataFrame
            Dataframe containing the enrichment p-value results that will be collapsed to maximize information being presented

    Returns:
    --------
        - ngrams_2_collapse: list
            The n-grams that will be collapsed from the inputted list.
    
    '''
    
    # Defining potential collapsing n-gram families
    potential_collapse = {}
    for ngram in ngram_list:
        for inner_ngram in ngram_list:
            # Check for parent-child relationship
            if inner_ngram != ngram and ngram in inner_ngram:

                # Add to the dictionary an empty list if the n-gram is not present
                if ngram not in potential_collapse:
                    potential_collapse[ngram] = []
                potential_collapse[ngram].append(inner_ngram)


    # Now for each of these checking the FPR and p-values to see if they should be collapsed
    ngrams_2_collapse = set()
    for ngram, children in potential_collapse.items():

        # Get the parent n-grams values
        parent_p = res_df[res_df['ngram'] == ngram]['p'].tolist()
        parent_fpr = res_df[res_df['ngram'] == ngram]['FPR <= 0.05'].tolist()
        parent_cond = res_df[res_df['ngram'] == ngram]['variable'].tolist()
        
        # Check if it is only within 1 condition
        if len(parent_p) == 1:
            for child in children:
                child_p = res_df[res_df['ngram'] == child]['p'].tolist()
                child_fpr = res_df[res_df['ngram'] == child]['FPR <= 0.05'].tolist()
                child_cond = res_df[res_df['ngram'] == child]['variable'].tolist()
                
                # Only collapse if they have the same number of conditions and match
                if len(child_p) == 1:
                    if child_cond == parent_cond:
                        # If the child one is more signficant and FPR values are not the same then keep it otherwise collapse it
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
                    # Checking both the p-vals as showing similar trends and child n-grams are not more signficant then keep
                    if any(c < p for c,p in zip(child_p,parent_p)) and any(p != c for c,p in zip(child_fpr,parent_fpr)):
                        pass
                    elif any(p != c for c,p in zip(child_fpr,parent_fpr)):
                        pass
                    else:
                        ngrams_2_collapse.add(child)
    
    return ngrams_2_collapse

def gather_enrichment_results(hyper_values, fpr_values):
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
    res['FPR <= 0.05'] = res['FPR'] <= 0.05
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


def plot_functional_scores(res, show_FPR_handle=True, aspect = 0.9, order = None):
    '''
    This creates the bubble plots for both the separation and distinction scores calculated by deDANSy.

    Parameters:
    -----------
        - res: pandas DataFrame
            The dataframe containing all scores and FPR values of the deDANSy analysis
        - show_FPR_handle: bool
            Whether the FPR portion of the legend should be displayed
        - aspect: float
            The aspect ratio of both plots
        - order: dict
            Key-value pairs of comparisons and their order

    Returns:
    --------
        - ax: list of matplotlib Axes
            The axes of each subplot generated

    '''

    # Starting with the Separation Scores subplot
    data_plot, comp_order = create_score_plot_data(res, 'Separation', order)
    
    _, axs = plt.subplots(1,2)
    plt.subplot(1,2,1)
    sns.scatterplot(data_plot, x='Separation_Category_Order', y = 'Order',
                    size='Separation_Score',
                    hue='Separation_Significance',
                    sizes = (1,50), size_norm = (0,5),
                    palette=['mediumorchid', 'silver'],
                    hue_order=[True, False],
                    linewidth = 0.5,edgecolor='k')

    # Adding in labels
    plt.title('Separation Score')
    plt.xlabel(None)
    plt.ylabel(None)
    
    # Cleaning up the ticks
    plt.xticks([0,1],['More', 'Less'], rotation=45, ha='right')
    plt.yticks([v for v in comp_order.values()], [k for k in comp_order.keys()])
    plt.xlim(-0.5,1.5)
    
    # Cleaning up the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = clean_up_legend(handles, labels, show_FPR_handle)
    l = plt.legend(new_handles,new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1)
    l.get_frame().set_linewidth(0.5)

    # Setting some aesthetics for the handles
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.5)
        
    # Grid if there are more than 3 comparisons to provide a slight visual guidance
    if len(set(data_plot['Comparison'].tolist())) >= 3:
        plt.grid(visible=True,axis='y', linewidth =0.25, linestyle= ':')
    plt.gca().set_aspect(aspect)

    # Now the Distinction Score
    data_plot,comp_order = create_score_plot_data(res, 'Distinction', order)
    plt.subplot(1,2,2)
    sns.scatterplot(data_plot,x = 'Distinction_Category_Order',
                    y = 'Order', size='Distinction_Score',hue='Distinction_Significance',sizes = (1,50),size_norm=(0,5),
                    palette=['seagreen', 'silver'], hue_order=[True, False],
                    linewidth = 0.5,edgecolor='k')
    
    # Tick clean up
    plt.xticks([0,1],['Stably Distinct', 'Unstable/Overlap'], rotation=45, ha='right')
    plt.yticks([v for v in comp_order.values()], [k for k in comp_order.keys()])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('Distinction Scores',fontdict={'size':6})
    plt.xlim([-0.5,1.5])
    plt.gca().set_aspect(aspect)
    plt.ylabel(None)
    plt.tick_params('y',labelleft=None)

    # Legend clean up and formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles, new_labels = clean_up_legend(handles, labels, show_FPR_handle)
    l = plt.legend(new_handles,new_labels,bbox_to_anchor=(1,1), edgecolor='k', handletextpad=0.1)
    l.get_frame().set_linewidth(0.5)
    
    # Aesthetics of specific legend handles
    for h in l.legend_handles:
        if not isinstance(h, lines.Line2D):
            h.set_edgecolor('k')
            h.set_linewidth(.5)
    
    # Grid if there are more than 3 comparisons to provide a slight visual guidance
    if len(set(data_plot['Comparison'].tolist())) >=3:
        plt.grid(visible=True,axis='y', linewidth =0.25, linestyle= ':')
        
    return axs

def clean_up_legend(handles, labels, show_FPR = True):
    '''
    An internal function that cleans up the legend of the bubble plot to ensure clear communication of results and scores. If desired the FPR portion of the legend will be omitted if it does not provide useful information.

    Parameters:
    -----------
        - handles: list
            List of handles for the unmodified legend of the matplotlib/seaborn plot
        - labels: list
            List of labels for the unmodified legend of the matplotlib/seaborn plot
        - show_FPR: bool
            Flag of whether the FPR portion of the legend should be displayed.
    
    Returns:
    --------
        - new_h: list
            List of handles for the new legend of the matplotlib/seaborn plot
        - new_l: list
            List of labels for the new legend of the matplotlib/seaborn plot
    '''
    if show_FPR:
            handles_2_rm = [0,5,7,9]
    else:
        handles_2_rm = [0,1,2,5,7,9] #Will not always need the FPR legend details so will remove them as well
    
    new_h = [h for i,h in enumerate(handles) if i not in handles_2_rm]
    new_l = [h for i,h in enumerate(labels) if i not in handles_2_rm]
    
    if show_FPR:
        new_l[0] = 'FPR$\leq$0.05'
        new_l[1] = 'FPR > 0.05'
        new_l[2] = 'Score'
    else:
        new_l[0] = 'Score'

    return new_h, new_l

def create_score_plot_data(data, metric, order = None):
    '''
    This creates the plot data for generating the final bubble plot for a multicomparison result. This ensures a consistent easy to read bubble size is present and creates the order the plot will be generated in. 

    Parameters:
    -----------
        - data: pandas DataFrame
            The scores dataframe generated by the deDANSy object
        - metric: str
            Either the Separation or Distinction score that will be used for plotting
        - order: dict (Optional)
            Key-value pairs that determine which order the multiple comparisons will be displayed in with the keys being the comparison and the values the order.
    
    Returns:
    --------
        - plot_data: pandas DataFrame
            A modified dataframe which contains a new Order column and additional values to create consistent sizing
        - comp_map: dict
            The order dictionary that will be used for displaying the comparisons
    '''
    plot_data = data.copy()

    if order is None:
        comps = sorted(plot_data['Comparison'].dropna().tolist())
        comp_map = {v:i for i,v in enumerate(comps)}
    else:
        comp_map = order
    plot_data['Order'] = plot_data['Comparison'].map(comp_map)

    if metric == 'Separation':
        max_vals = np.ceil(plot_data['Separation_Score'].max())
        cat_order = {'More':0,'Less':1}
        plot_data['Separation_Category_Order'] = plot_data['Separation_Category'].map(cat_order)
        for i,v in enumerate(np.linspace(0,max_vals,5)):
            plot_data.loc[i+len(plot_data)] = None
            plot_data.loc[i+len(plot_data), 'Separation_Score'] = v
    else:
        # The IQR scores tend to be between 0-2 so getting the closest integer and then taking 5 steps to force the size
        max_vals = np.ceil(plot_data['Distinction_Score'].max())
        cat_order = {'Stably Distinct':0,'Unstable/Overlapping':1}
        plot_data['Distinction_Category_Order'] = plot_data['Distinction_Category'].map(cat_order)
        for i,v in enumerate(np.linspace(0,max_vals,5)):
            plot_data.loc[i+len(plot_data)] = None
            plot_data.loc[i+len(plot_data), 'Distinction_Score'] = v

    return plot_data,comp_map