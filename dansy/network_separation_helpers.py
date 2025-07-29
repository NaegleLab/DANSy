import warnings
import networkx as nx
import numpy as np
from dansy.dansy import dansy

def mean_minimum_dist(G_nodes, ref_net_params_dict):
    '''
    Calculates the average shortest path length between nodes within the network along a reference network. Nodes that are isolates within the reference network have a shortest path length of 0, and nodes within the network which represent a completely collapses connected component also have a shortest path length of 0 or a penalty value if provided. 
    '''
    # Unpacking the reference information that is used in calculating the shortest path lengths
    ref_isolates = ref_net_params_dict['isolates']

    if isinstance(G_nodes, nx.Graph):
        nodesOI = set(G_nodes.nodes())
    else:
        nodesOI = G_nodes

    if len(nodesOI) > 0:
        # Don't bother checking the expected isolates. These will be accounted for in the total length as they contribute as 0s in the final calculation.
        nodes_2_check = nodesOI.difference(ref_isolates)
        
        # Get the shortest path length for individual nodes that do exist in the reference network
        spl = spl_inner(nodes_2_check, nodesOI, ref_net_params_dict)
        
        # Ensuring the isolates are accounted for in the average
        mean_min_dist = np.sum(spl)/len(nodesOI)
    
    else:
        mean_min_dist = np.nan

    return mean_min_dist

def spl_inner(nodes_2_check, all_nodes, ref_net_analysis_parameters):
    '''
    An inner calculation function that gets the shortest path length between nodes of interest and target nodes that are often found within 2 subnetworks of a larger network. For nodes that do not have connections between the subnetworks these are given a penalty term that is defined in the reference network information provided.

    Parameters:
    -----------
        - nodes_2_check: list
            - Nodes of interest who act as a source of the shortest path lengths are to be gathered
        - all_nodes: list
            - The target list of nodes that the shortest path distance from the nodes of interest is to be queried for.
        - ref_net_analysis_parameters: dict
            - A dictionary that contains several components of the reference, background network that are used in the calculation. See build_network_reference_dict for more details.

    Returns:
    --------
        - spl: list
            - The shortest path distances for all the nodes of interest. (Note that if there are no connections between a node of interest and the target nodes, this will be the penalty term found in the reference network dict.)
    
    '''
    
    reference_spl_dict = ref_net_analysis_parameters['spl']
    penalty_input = ref_net_analysis_parameters['penalty']

    # Verify that the penalty that was to be used is an appropriate format and can be used for downstream analysis
    dynamic_flag = False
    # If a constant term is provided
    if isinstance(penalty_input, int):
        penalty = penalty_input
    # If one of the acceptable methods is provided and thus being calculated
    elif penalty_input == 'dynamic':
        dynamic_flag = True
    else:
        raise TypeError('Inappropriate penalty term input')

    # Some precomputing for the dynamic method
    if dynamic_flag:
    
        if 'CCs' in ref_net_analysis_parameters and 'cc_diameters' in ref_net_analysis_parameters:
            ccs = ref_net_analysis_parameters['CCs']
            cc_diams = ref_net_analysis_parameters['cc_diameters']
        else:
           raise ValueError('An invalid reference data dictionary was provided. Please run build_network_reference_dict')
        
        # Reduce the connected components that are being referenced are only the ones that contain nodes within the network.
        ccs = [cc for cc in ccs if cc.intersection(nodes_2_check)]
        
        # Fail-safe check for debugging
        if nodes_2_check:
            if (not cc_diams) & (len(ccs) > 0):      
                raise ValueError(f'Whoops something went very wrong and this should not happen.')
    
    # In some instances there may be some nodes removed from the spl dict (via pruning), so will filter out nodes that are not a part of that
    orig_nodes = len(nodes_2_check)
    nodes_2_check = set(nodes_2_check).intersection(reference_spl_dict.keys())

    # If there was a pruned shortest path length dict passed then ensuring that the full complement of nodes does not include the pruned nodes
    if len(nodes_2_check) != orig_nodes:
        all_nodes = set(all_nodes).intersection(reference_spl_dict.keys())


    # Now getting the shortest path lengths based on the provided parameters
    spl = []
    for node in nodes_2_check:
        tmp_spl = reference_spl_dict[node]
        spl_dict = {k:v for k,v in tmp_spl.items() if k in all_nodes}
        if node in spl_dict:
            del spl_dict[node] # Removing the self-reference that is always 0

        if spl_dict:
            spl.append(min(spl_dict.values()))
        else: # If a node by itself represents a single connected component from the reference adding a penalty for collapsing the entire connected component
            
            # The dynamic method 
            if dynamic_flag:
                penalty = cc_diams[node] + 1
                spl.append(penalty)
            else:
                spl.append(penalty)

    return spl


def network_separation(G_in, H_in, ref_G_data, mmd_verbose = False, force_run = False, verbose = True):
    '''
    Calculates the network separation between two networks of interest that lie on a common larger, reference network.

    Parameters:
    -----------
        - G_in, H_in: networkx Graph | DomainNgramNetwork
            - networks of interest
        - ref_G_data: dict
            - dict containing information about the larger, common network that the two given networks are subnetworks of.
        - mmd_verbose: bool (Optional, to be deprecated)
            - Whether a statement is printed.
        - force_run: bool (Optional)
            - Whether the network separation should be run even if it does not meet minimum values necessary.
    
    Returns:
    --------
        - s : float
            - the network separation of two different subnetworks.

    '''
    # Unpacking the reference network information that is used for getting minimum distance values
    ref_isolates = ref_G_data['isolates']

    # Now getting the minimum distances for both the sub-networks of interest
    # As part of the mean minimum distance calculation there is a secondary option to set penalties for disconnected nodes. If it is not provided then will set the penalty to zero
    mmd_ref = {**ref_G_data} # Need to create a copy otherwise can overwrite the penalty value by accident
    if 'mean_dist_penalty' in ref_G_data:
        if isinstance(ref_G_data['mean_dist_penalty'], int): # To allow a separate penalty term for the mean shortest path distance calculation from the network separation
            mmd_ref['penalty'] = ref_G_data['mean_dist_penalty']
        elif isinstance(ref_G_data['mean_dist_penalty'], bool):
            if ref_G_data['mean_dist_penalty']:
                pass # Already has the penalty term passed
            else:
                mmd_ref['penalty'] = 0     
        else:
            raise ValueError('An improper penalty value was provided for calculating the mean minimum distance.')   
    else:
        mmd_ref['penalty'] = 0

    # After some parameter sweeps there is a rough cutoff for networks that represent ~20 proteins that tend to exhibit wide ranges in network separation values that are not as informative to network characteristics as they frequently will have larger isolate fractions than landing within a large connected component. Here checking that both networks meet the minimum size otherwise raising an error. (Unless the force run flag is present.)
    if isinstance(G_in, dansy) and isinstance(H_in, dansy):  
        if (len(G_in.protsOI) < 20) or (len(H_in.protsOI) < 20):
            if force_run:
                if verbose: 
                    warnings.warn('At least one network does not reach recommended minimum size, but will still be analyzed.')
            else:
                raise ValueError('At least one network does not reach recommended minimum size.')
            
        G = G_in.G
        H = H_in.G
    else:
        G = G_in
        H = H_in

    # Getting the list/set of nodes for doing calculations on
    G_v = set(G.nodes())
    H_v = set(H.nodes())

    if mmd_verbose: # This is for debugging purposes but can be retained if needed.
        print(f"Input penalty for the mean minimum distance is {mmd_ref['penalty']}")
    G1_mmd = mean_minimum_dist(G_v,mmd_ref)
    G2_mmd = mean_minimum_dist(H_v,mmd_ref)

    # To find the separation between the two networks will find the minimum distance from nodes in one network to nodes in the other network.
    # First finding the overlap in nodes as these will be 0 and don't need to go through the process.
    node_overlap = G_v.intersection(H_v)
    nodesOI_1 = G_v.difference(node_overlap)
    nodesOI_2 = H_v.difference(node_overlap)

    # For the nodes that were isolates in the reference network removing them from running in the for loop as they will not provide any additional information.
    nodesOI_1 = nodesOI_1.difference(ref_isolates)
    nodesOI_2 = nodesOI_2.difference(ref_isolates)
    
    # Running through the non-isolate nodes and getting the shortest distances from nodes within one network to nodes in another.
    spl_union = spl_inner(nodesOI_1, H_v,ref_G_data)
    tmp = spl_inner(nodesOI_2,G_v,ref_G_data)
    spl_union += tmp

    # Now for the isolates to account for their contributions and creating a constant 1 path length penalty for those that are not shared between networks
    isols_1 = G_v.intersection(ref_isolates)
    isols_2 = H_v.intersection(ref_isolates)
    isol_correction_1 = len(isols_1.difference(isols_2))
    isol_correction_2 = len(isols_2.difference(isols_1))

    # Getting the average minimum distance across the union of both networks
    union_mmd = (sum(spl_union)+isol_correction_1+isol_correction_2)/(G.number_of_nodes()+H.number_of_nodes())
    s = union_mmd - (G1_mmd+G2_mmd)/2

    return s   

 
def build_network_reference_dict(ref_ngram_net, penalty = None):
    '''
    This builds the dict that contains the reference network information necessary for calculating the network separation value from a provided DomainNgramNetwork object. 
    
    It is recommended to use this function to generate the reference dictionary prior to calculating network separation, especially when using a dynamic penalty and comparing several networks, to improve the execution speed of the calculation.

    Parameters:
    -----------
        dnn: DomainNgramNetwork
            - A populated Domain N-gram network that will be used as a reference
        penalty: str or int (Optional)
            - What type of penalty will be used for network separation. If not specified will default to the dynamic method.

    Returns:
    --------
        ref_data_dict: dict
            - Key-value pairs that are used for network separation calculations.


    Key-Value Pairs:
        - penalty: int | 'dynamic'
            The penalty term that will be used for shortest path distances
        - spl: dict
            The shortest path length of all pairs of nodes within the network.
        - isolates: list
            List of isolates in the network
        - CCs: list
            List of tuples of nodes that are the connected components of the network
        - cc_diameters: dict
            Key-value pairs of nodes and the diameter of the connected componenet they are a part of.
        - G: networkx Graph
            The graph of the full network.

    '''

    # Default value for the penalty term
    if penalty == None:
        penalty = 'dynamic'

    # Shortest path lengths between individual nodes and non-isolate nodes
    comp_pw_shortest_path_lens = dict(nx.all_pairs_shortest_path_length(ref_ngram_net.G))
    ccs = nx.connected_components(ref_ngram_net.G)
    ccs = [cc for cc in ccs if len(cc) > 1]
    
    # Building a diameter dictionary where the keys are each node and the value is the diameter for the connected component including that node
    cc_diams = {}
    for cc in ccs:
        d = nx.diameter(ref_ngram_net.G.subgraph(cc))
        for node in cc:
            cc_diams[node] = d

    ref_data_dict = {'spl':comp_pw_shortest_path_lens,
                    'penalty':penalty,
                    'isolates':list(nx.isolates(ref_ngram_net.G)),
                    'CCs':ccs,
                    'cc_diameters':cc_diams,
                    'G':ref_ngram_net.G}

    return ref_data_dict
