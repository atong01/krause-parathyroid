import numpy as np
import phate
import os
import scprep
import magic
import pickle
import graphtools
import shutil
import gzip


def get_common_genes(sample_list):
    common_set = set(sample_list[0].columns.values.tolist())
    for s in sample_list[1:]:
        common_set = common_set.intersection(set(s.columns.values.tolist()))
    return list(common_set)

data_name = "PPA"

SAMPLE_NAMES = ['Y7', 'Y9', 'Y11', 'Y13']
SEED = 42
samples = []
s = scprep.io.load_10X_HDF5(os.path.join(
        os.path.abspath(os.sep),
        "data", "lab", "DataSets",
        "Krause_2018_primary_parathyroid_adenoma",
        "paray7_HHT_cellranger",
        "filtered_gene_bc_matrices_h5.h5",
    ), gene_labels='both', allow_duplicates=True)
samples.append(s)
s = scprep.io.load_10X_HDF5(os.path.join(
        os.path.abspath(os.sep),
        "data", "lab", "DataSets",
        "Krause_2018_primary_parathyroid_adenoma",
        "ParaY9_HHT_cellranger",
        "filtered_gene_bc_matrices_h5.h5",
    ), gene_labels='both', allow_duplicates=True)
samples.append(s)
for sample in SAMPLE_NAMES[2:]:
    s = scprep.io.load_10X(os.path.join(
        os.path.abspath(os.sep),
        "data", "lab", "DataSets",
        "Krause_2019_primary_parathyroid_adenoma",
        "%s_HHT_cellranger" % sample,
        "filtered_feature_bc_matrix",
    ), gene_labels='both', allow_duplicates=True)
    # s = scprep.filter.remove_rare_genes(s, min_cells=3)
    samples.append(s)
    
common_genes = get_common_genes(samples)
samples = [s[common_genes] for s in samples]
samples_norm = [scprep.normalize.library_size_normalize(s, rescale=10000) for s in samples]
samples_sqrt_norm = [scprep.transform.sqrt(s) for s in samples]
data, sample_labels = scprep.utils.combine_batches(samples_sqrt_norm, SAMPLE_NAMES)
data.astype('float32')

mnn_graph = graphtools.Graph(data,
                             sample_idx=sample_labels,
                             n_pca=100, 
                             random_state=42)

phate_op = phate.PHATE(n_components=3,
                       random_state=42,
                       knn_dist='precomputed', n_jobs=36, gamma=0)
mnn_data_phate_3d = phate_op.fit_transform(mnn_graph.K)
phate_op.set_params(n_components=2)
mnn_data_phate_2d = phate_op.fit_transform(mnn_graph.K)
np.save("{}Phate2d.npy".format(data_name), mnn_data_phate_2d)
np.save("{}Phate3d.npy".format(data_name), mnn_data_phate_3d)

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

# The largest component is mg.X (>90% in my usage) (atong)
mg.X = mg.X.astype('float32')

with open('magic.pickle', 'wb') as handle:
    pickle.dump(mg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('magic.pickle', 'rb') as f_in, gzip.open('magic.pickle.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
