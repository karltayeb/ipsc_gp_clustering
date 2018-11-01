import numpy as np
import pandas as pd

import sys
import pickle
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import numpy as np
from MixtureSVGP import MixtureSVGP

def readgenesets(filename):
    genesets={}
    with open(filename,'r') as fh:
        for line in fh:
            eachline = line.strip('\n').split('\t')
            #print(len(eachline[2::]))
            #if(len(eachline[2::]) >= 30 and len(eachline[2::]) <= 500):
            genesets[eachline[0]] = eachline[2::]
    return genesets


def gsea(testlist, background, gene_set):
    """ Fisher's exact test for enrichment"""
    # Returns raw p-value
    from scipy.stats import fisher_exact
    from collections import defaultdict, OrderedDict
    gene_sets = readgenesets(gene_set)
    size_gene_sets = len(gene_sets)
    enrich_out = OrderedDict()
    for i in gene_sets:
        # if len(gene_sets[i]) >= 50 and len(gene_sets[i]) <= 500:
        test_inset = len(set(testlist).intersection(gene_sets[i]))
        test_notinset = len(set(testlist)) - test_inset
        background_list = set(background) - set(testlist)
        background_inset = len(set(background_list).intersection(gene_sets[i]))
        background_notinset = len(set(background_list)) - background_inset
        oddsratio, pvalue = fisher_exact([[test_inset,background_inset],[test_notinset,background_notinset]],alternative='greater')
        bf_adjusted = pvalue * size_gene_sets
        bf_adjusted = min(bf_adjusted,1)
        enrich_out[i] = tuple([test_inset, test_notinset, background_inset, background_notinset, oddsratio, pvalue, bf_adjusted, ','.join(set(testlist).intersection(gene_sets[i]))])
    return enrich_out


def run_gsea(scores, n, gene_set_path='../gsea/data/h.all.v5.1.symbols.gmt.txt'):
    ranked_abosolute = np.array(scores.abs().sort_values(
        ascending=False).iloc[:n].index)
    ranked_largest = np.array(scores.sort_values(
        ascending=False).iloc[:n].index)
    ranked_smallest = np.array(scores.sort_values(
        ascending=True).iloc[:n].index)

    background = np.array(scores.index)

    index = ['test_inset', 'test_not_inset', 'background_inset',
             'background_not_inset', 'oddsratio', 'pvalue',
             'bonferonni-adjusted', 'genes']

    absolute = pd.DataFrame(gsea(
        ranked_abosolute, background, gene_set_path), index=index)
    largest = pd.DataFrame(gsea(
        ranked_largest, background, gene_set_path), index=index)
    smallest = pd.DataFrame(gsea(
        ranked_smallest, background, gene_set_path), index=index)

    return pd.concat({'absolute': absolute,
                     'group 1': largest, 'group 2': smallest})


def gen_gsea_df(assignments, pickle_path=None,
                gene_set_path='../gsea/data/h.all.v5.1.symbols.gmt.txt',
                threshold=1e-2, rerun=False):
    N, K = assignments.Phi.shape
    G, L = assignments.Lambda.shape

    if not rerun:
        try:
            results = pd.read_pickle(pickle_path)
            return results
        except:
            pass

    normalized_data_df, _, _ = load_data(
        './data/quantile_normalized_no_projection.txt')
    normalized_data_df = normalized_data_df.iloc[:, :G]

    lines = np.array(normalized_data_df.index.levels[0])
    sample_assignments = assignments.Phi.argmax(axis=1)

    gene_dict = pickle.load(open('./data/ensembl_to_gene_name', 'rb'))
    gene_names = np.array(normalized_data_df.columns)
    gene_names2 = np.array([gene_dict[gene] for gene in gene_names])

    cell_line_grp1 = lines[sample_assignments == 0]
    cell_line_grp2 = lines[sample_assignments == 1]

    cell_line_grp1, cell_line_grp2

    gene_assignments = assignments.Lambda.argmax(axis=1)
    index = ['test_inset', 'test_not_inset', 'background_inset',
             'background_not_inset', 'oddsratio', 'pvalue',
             'bonferonni-adjusted', 'genes']
    results = pd.concat(
        {l: pd.DataFrame(gsea(
            gene_names2[gene_assignments == l],
            gene_names2, gene_set_path), index=index) for l in range(L)})
    results.to_pickle(pickle_path)
    return results


def load_data(data_path):
    data_df = pd.read_csv(data_path, delim_whitespace=True, index_col=0)
    data = data_df.as_matrix()

    data = data - data.mean(axis=1)[:, np.newaxis]
    data = data / data.std(axis=1)[:, np.newaxis]

    normalized_data_df = data_df
    normalized_data_df.loc[:, :] = data
    normalized_data_df = normalized_data_df.T
    normalized_data_df = normalized_data_df.reset_index()

    normalized_data_df['line'] = normalized_data_df['index'].apply(
        lambda x: x.split('_')[0])
    normalized_data_df['time'] = normalized_data_df['index'].apply(
        lambda x: x.split('_')[1])
    normalized_data_df['time'] = normalized_data_df['time'].astype(int)
    normalized_data_df = normalized_data_df.drop('index', axis=1)

    data_dict = {}
    for l, grp in normalized_data_df.groupby('line'):
        grp = grp.sort_values('time')
        grp = grp.drop('line', axis=1)
        grp = grp.set_index('time')
        data_dict[l] = grp
    normalized_data_df = normalized_data_df.set_index(['line', 'time'])

    n_dim = normalized_data_df.shape[1]
    nans = [np.nan] * n_dim
    X = [[data_dict[key].loc[i].tolist() if (i in data_dict[key].index)
          else nans for i in range(16)] for key in data_dict.keys()]
    X = np.array(X)

    return normalized_data_df, X, data_dict


def reload_model(assignments, model_params, kernel_type, T=16):
    N, K = assignments.Phi.shape
    G, L = assignments.Lambda.shape


    minibatch_size = None
    
    ###############
    # Make models #
    ###############

    num_clusters = K * L

    print(num_clusters)
    if kernel_type is 'shared':
        kernel = mk.SharedIndependentMok(gpflow.kernels.RBF(1), num_clusters)
    elif kernel_type is 'seperate':
        kernel = mk.SeparateIndependentMok(
            [gpflow.kernels.RBF(1) for _ in range(num_clusters)])

    feature = mf.SharedIndependentMof(
        gpflow.features.InducingPoints(np.arange(T).astype(
            np.float64).reshape(-1, 1)))
    likelihood = gpflow.likelihoods.Gaussian()

    # model -- for hyperparameter learning
    with gpflow.defer_build():
        m = MixtureSVGP(np.zeros((1, 1)).astype(np.float64),
                        np.zeros((1, 1)).astype(np.float64),
                        np.zeros((1, 1)).astype(np.int32),
                        kern=kernel,
                        num_latent=num_clusters, num_data=1,
                        likelihood=likelihood,
                        feat=feature, minibatch_size=minibatch_size)

        # only train hyperparams
        for param in m.parameters:
            param.trainable = False

    # compile models
    m.compile()
    return m