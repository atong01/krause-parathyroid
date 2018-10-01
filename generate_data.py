import numpy as np
import phate
import os
import scprep
import magic
import pickle
import graphtools

data_name = "PPA"

paray9 = scprep.io.load_10X_HDF5(os.path.join(
    os.path.abspath(os.sep),
    "data", "lab", "DataSets",
    "Krause_2018_primary_parathyroid_adenoma",
    "ParaY9_HHT_cellranger",
    "filtered_gene_bc_matrices_h5.h5",  # raw_gene_bc_matrices
), gene_labels='both', allow_duplicates=True)
paray9 = scprep.filter.remove_rare_genes(paray9, min_cells=3)

paray9_norm = scprep.normalize.library_size_normalize(paray9)
paray9_norm = scprep.transform.sqrt(paray9_norm)

paray7 = scprep.io.load_10X(os.path.join(
    os.path.abspath(os.sep),
    "data", "lab", "DataSets",
    "Krause_2018_primary_parathyroid_adenoma",
    "paray7_HHT_cellranger",
    "filtered_gene_bc_matrices",  # raw_gene_bc_matrices
    "GRCh38"), gene_labels='both')
paray7 = scprep.filter.remove_rare_genes(paray7, min_cells=3)

paray7_norm = scprep.normalize.library_size_normalize(
    paray7, rescale=paray9_norm.iloc[0].sum())
paray7_norm = scprep.transform.sqrt(paray7_norm)

common_genes = set(paray7_norm.columns.values.tolist())
common_genes = common_genes.intersection(paray9_norm.columns.values.tolist())
data, sample_labels = scprep.utils.combine_batches(
    [paray7_norm[list(common_genes)], paray9_norm[list(common_genes)]],
    ['Y7', 'Y9'])

mnn_graph = graphtools.Graph(data, sample_idx=sample_labels, n_pca=100, knn=5,
                             random_state=42,
                             decay=15, kernel_symm='theta', theta=0.99)
pth_idx = (data[scprep.utils.get_gene_set(
    data, starts_with="PTH ").values] >= 11).values.flatten()


ph = phate.PHATE(n_components=2,
                 random_state=42,
                 knn_dist='precomputed')
phate_data = ph.fit_transform(mnn_graph.kernel)
np.save("{}Phate2d.npy".format(data_name), phate_data[pth_idx])

ph.set_params(n_components=3)
phate3_data = ph.transform()
np.save("{}Phate3d.npy".format(data_name), phate3_data[pth_idx])

mnn_graph = graphtools.Graph(data.iloc[pth_idx],
                             sample_idx=sample_labels[pth_idx],
                             n_pca=100, knn=5,
                             random_state=42,
                             decay=15, kernel_symm='theta', theta=0.99)
mg = magic.MAGIC(
    random_state=42,
    a=mnn_graph.decay, k=mnn_graph.knn - 1, n_pca=mnn_graph.n_pca)
data_magic = mg.fit_transform(data, graph=mnn_graph)
_ = mg.fit_transform(data)

# reduce memory footprint
del mg.graph.data
del mg.graph.data_nu
del mg.graph._kernel
del mg.graph._diff_op
del mg.graph.subgraphs
del mg.graph.sample_idx

with open('magic.pickle', 'wb') as handle:
    pickle.dump(mg, handle, protocol=pickle.HIGHEST_PROTOCOL)
