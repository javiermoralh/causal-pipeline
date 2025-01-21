import torch
import numpy as np
import pandas as pd
import graphviz
import itertools
from itertools import combinations
import networkx as nx

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from castle.common.priori_knowledge import PrioriKnowledge
from dowhy import gcm
from dowhy.gcm.independence_test import approx_kernel_based
from pgmpy.base import DAG as PGMDAG


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph


def get_testable_implications(graph):
    """
    Function to get all testable implications
    """
    all_nodes = list(graph.nodes)
    all_possible_combinations = list(
        combinations(all_nodes, 2)
    )  # Generating sets of indices of size 2 for different x and y
    simple_independence = []
    conditional_independences = []
    for combination in all_possible_combinations:  # Iterate over the unique 2-sized sets [x,y]
        a = combination[0]
        b = combination[1]
        try: # we try path a->b or b->a
            all_paths = list(nx.all_shortest_paths(graph, source=a, target=b))
        except nx.exception.NetworkXNoPath:
            try:
                all_paths = list(nx.all_shortest_paths(graph, source=b, target=a))
            except:
                all_paths = []
        if len(all_paths) == 0:
            k_set = {}
        else:
            k_set = {n for n in all_paths[0] if n not in [a, b]}

        if len(k_set) == 0 and len(all_paths) == 0:
            simple_independence.append((a, b))
        if len(k_set) > 0:
            conditional_independences.append((a, b, k_set))
    return simple_independence, conditional_independences


def get_nx_graph_FCI(causal_graph_matrix, feature_names):
    """
    Function to get networkx DAG from causal-learn FCI output
    """
    # add edges from a list of tuples
    id_features = {i: c for i, c in enumerate(feature_names)}
    edges_list = []
    for index_1, feature_1 in id_features.items():
        for index_2, feature_2 in id_features.items():
            first_connection = causal_graph_matrix[index_1, index_2]
            second_connection = causal_graph_matrix[index_2, index_1]
            if (first_connection == -1) & (second_connection == 1):
                causal_relations_feature = [(id_features[index_1], id_features[index_2])]
                edges_list.extend(causal_relations_feature)  
            if (first_connection == 1) & (second_connection == -1):
                causal_relations_feature = [(id_features[index_2], id_features[index_1])]
                edges_list.extend(causal_relations_feature)  
            if (first_connection == 2) & (second_connection == 1):
                causal_relations_feature = [(id_features[index_1], id_features[index_2])]
                edges_list.extend(causal_relations_feature)  
            if (first_connection == 1) & (second_connection == 2):
                causal_relations_feature = [(id_features[index_2], id_features[index_1])]
                edges_list.extend(causal_relations_feature)
            if ((first_connection == 1) & (second_connection == 1)) | ((first_connection == 2) & (second_connection == 2)):
                causal_relations_feature = [(id_features[index_1], id_features[index_2])]
                edges_list.extend(causal_relations_feature)  
                causal_relations_feature = [(id_features[index_2], id_features[index_1])]
                edges_list.extend(causal_relations_feature)  
            

    # create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges_list)
    return G


def create_prior(data, package, TREATMENT, TARGET, algorithm=None, force_treatmemnt_output=False):
    """
    Function to create causal graph priors dependending on package and algorithm
    assumptions:
    
    -> No feature can be caused by treatment except outcome
    -> No feature can be casued by outcome
    -> Some sociodemo features like age are not caused by other features
    """
    n_dim = data.shape[1]
    feature_ids = {c: i for i, c in enumerate(data.columns)}
    id_treatment = feature_ids[TREATMENT]
    id_outcome = feature_ids[TARGET]
    
    if package == "causal-learn":
        priori = BackgroundKnowledge()
        for feature, index in feature_ids.items():
            node, node_tretament, node_outcome = (
                GraphNode('X' + str(index+1)),
                GraphNode('X' + str(id_treatment+1)),
                GraphNode('X' + str(id_outcome+1)),
            )
            priori.add_forbidden_by_node(node_outcome, node_tretament)
            if feature not in [TREATMENT, TARGET]:
                priori.add_forbidden_by_node(node_tretament, node)
                priori.add_forbidden_by_node(node_outcome, node)
            if "age" in feature_ids.keys():
                node_age = GraphNode('X' + str(feature_ids["age"]+1))
                priori.add_forbidden_by_node(node, node_age)
        if force_treatmemnt_output:
            priori.add_required_by_node(node_tretament, node_outcome)
    elif package == "gcastle":
        if algorithm == "pc":
            priori = PrioriKnowledge(n_dim)
            forbidden_edges = [(id_outcome, id_treatment)]
            for feature, index in feature_ids.items():
                if feature not in [TREATMENT, TARGET]:
                    forbidden_edges.extend([(id_treatment, index), (id_outcome, index)])
                if "age" in feature_ids.keys():
                    forbidden_edges.extend([(index, feature_ids["age"])])
            priori.add_forbidden_edges(forbidden_edges)
            if force_treatmemnt_output:
                priori.add_required_edge(id_treatment, id_outcome)
        elif algorithm == "lingam":
            # define a matrix full of -1
            priori = np.full((n_dim, n_dim), -1)
            priori[id_outcome, id_treatment] = 0
            for feature, index in feature_ids.items():
                if feature not in [TREATMENT, TARGET]:
                    priori[id_treatment, index] = 0
                    priori[id_outcome, index] = 0
                if "age" in feature_ids.keys():
                    priori[index, feature_ids["age"]] = 0
    return priori


# Find all cycles in the graph
def find_cycles(G):
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print("Found cycles:")
            for cycle in cycles:
                print(" -> ".join(cycle + [cycle[0]]))  # Add first node at end to complete cycle
            return cycles
        else:
            print("No cycles found - graph is a DAG")
            return []
    except nx.NetworkXNonImplemented:
        print("Graph is undirected. Convert to directed graph first.")
        return []

# Check if graph is a DAG and find problematic edges
def analyze_dag_issues(G):
    if nx.is_directed_acyclic_graph(G):
        print("Graph is already a DAG")
        return []
    else:
        cycles = find_cycles(G)
        problematic_edges = []
        
        # Extract edges from cycles
        for cycle in cycles:
            for i in range(len(cycle)):
                edge = (cycle[i], cycle[(i + 1) % len(cycle)])
                problematic_edges.append(edge)
                
        print("\nProblematic edges that form cycles:")
        for edge in set(problematic_edges):  # Use set to remove duplicates
            print(f"{edge[0]} -> {edge[1]}")
            
        return problematic_edges

# Remove cycles to make it a DAG
def make_dag(G):
    G_dag = G.copy()
    while not nx.is_directed_acyclic_graph(G_dag):
        cycles = list(nx.simple_cycles(G_dag))
        if not cycles:
            break
        # Remove first edge from first cycle found
        cycle = cycles[0]
        G_dag.remove_edge(cycle[0], cycle[1])
        print(f"Removed edge: {cycle[0]} -> {cycle[1]}")
    return G_dag


def get_edges_list_from_gcastlegraph(graph, features):
    # add edges from a list of tuples
    id_features = {i: c for i, c in enumerate(features)}
    causal_graph_matrix = graph.causal_matrix
    edges_list = []
    for index, feature in id_features.items():
        feature_edges = []
        conncetions = list((causal_graph_matrix[index] == 1).nonzero()[0])
        causal_relations_feature = [
            (id_features[index], id_features[c]) for c in conncetions
        ]
        edges_list.extend(causal_relations_feature)
    return edges_list

def get_outcome_only_causes(nx_g, outcome, treatment):
        # 1) Identify all direct parents of the outcome
    direct_parents_of_outcome = list(nx_g.predecessors(outcome))

    # 2) Among these direct parents, exclude any that also cause the treatment.
    features_only_cause_outcome = []
    for node in direct_parents_of_outcome:
        if node != treatment:
            if treatment not in nx.descendants(nx_g, node):
                features_only_cause_outcome.append(node)

    print("Features that cause ONLY the outcome:", features_only_cause_outcome)
    return features_only_cause_outcome
