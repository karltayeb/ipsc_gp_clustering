# ipsc_gp_clustering

## Assignments.py

Has assignment objects which keep track of gene and cell line level assignments.

pi [K] are cell line cluster mixture weights, the prior probability that a cell line belongs to a given cell line cluster
Phi [N, K] are variational parameters for latent cell line assignment. Phi[n, k] gives the (approximate) posterior probability that cell line n belongs to cell line cluster k

psi [L] are gene cluster mixture weights, the prior probability that a gene belongs to a given gene cluster
Lambda [G, L] are variational parameters for latent gene assignments. Phi[g, l] gives the (approximate) posterior probability that gene g belongs to gene cluster l

rho [2] a prior on whether a gene cluster is cell-line cluster specific or shared across cell line clusters.
Gamma [L, 2] are variational parameters for whether each gene cluster has a shared or cell line specific trajectory. Gamma[l, 0] = 1 indicates that gene cluster trajectories are cell-line cluster specific. Each cell line cluster is free to learn its own expression trajectory for gene cluster l. Gamma[l, 1]=1 indicates that gene cluster l should share an expression trajectory in all cell line clusters.

Each assignment object has a few key functions:
- entropy() computes the entropy of that variational distributions, necessary for computing the evidence lower bound (ELBO)
- compute_weights() computes and expanded representation of assignment. expands Phi, Lambda, and Gamma to get probability that a sample from (cell_line, gene) pair belongs to one of K*L clusters
- update_assignments(m, X, Y) closed form updates for assignment variables. m is the gaussian process model, X, Y are inpute and observations respectively. computes expected likelihood of data under the model, plus prior, appropriately normalized. Also updates phi and psi.


## MixtureSVGP.py

Gaussian process model implemented in GPFlow. The implementation is a simple extention of SVGP implemented in GPflow. We can think of this as K*L indpendent gaussian processes, where each (cell_line, gene) is weighted according to the probability that it comes from a particular (cell line cluster, gene cluster).

If we have N cell lines, G genes at T timepoints
X is [NGT, D] where D is the dimensionality of your inpiut
Y is [NGT, P] where P is the dimensionality of your output
weight_idx [NGT] integer indexing of the observations such that all samples belonging to the same (cell_line, gene) share an index.

## How to use

Input X, Y, weight_idx as above. We use gpflow's multioutput kernels to specify our model. Each output of the multioutput kernel corresponds to a (cell_line_cluster, gene_cluster) pair.

1. specify a kernel using gpflow multioutput kernel module.
2. specify a set of inducing points using gpflow multioutput feature module.
3. When initializing the model, be sure to specify num_latent, and if necessary minibatch size.

4. If data is only observed at a small number of input locations, and you are using a model with gaussian noise, we can speed up the computations for estimating the mean/variance by doing inference on a GP with the average observations as input. We can initialize a second model that we will feed a weighted average of our observations under the current assignment.

Learning proceeds as follows. We iterativelt use the first model to estimate kernel parameters, and the mean-model to estimate trajectories.
