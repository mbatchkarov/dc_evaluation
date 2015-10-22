import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from discoutils.thesaurus_loader import Vectors


def cluster_vectors(path_to_vectors, output_path, n_clusters=100, noise=0, n_jobs=4):
    vectors = Vectors.from_tsv(path_to_vectors, noise=noise)
    km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, random_state=0)
    clusters = km.fit_predict(vectors.matrix)
    num2word = np.array(vectors.row_names)
    idx = np.argsort(num2word)
    df = pd.DataFrame(dict(clusters=clusters[idx]), index=num2word[idx])
    df.to_hdf(output_path, key='clusters', complevel=9, complib='zlib')


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M')
    infile, outfile, num_cl, noise = sys.argv[1:]
    logging.info('Starting clustering with k=%r and noise %r', num_cl, noise)
    cluster_vectors(infile, outfile, int(num_cl), float(noise))
    logging.info('Done')
