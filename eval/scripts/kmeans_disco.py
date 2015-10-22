import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from discoutils.thesaurus_loader import Vectors


def cluster_vectors(path_to_vectors, output_path, n_clusters=100, noise=0, n_jobs=4):
    vectors = Vectors.from_tsv(path_to_vectors, noise=noise)
    km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, random_state=0, verbose=1)
    clusters = km.fit_predict(vectors.matrix)
    num2word = np.array(vectors.row_names)
    idx = np.argsort(num2word)
    df = pd.DataFrame(dict(clusters=clusters[idx]), index=num2word[idx])
    df.to_hdf(output_path, key='clusters', complevel=9, complib='zlib')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M')

    parser = argparse.ArgumentParser(description='Cluster distributional representations with k-means')
    parser.add_argument('--input', help='Path to input vectors file')
    parser.add_argument('--output', help='Path to output clusters file')
    parser.add_argument('--num-clusters', type=int, help='Number of clusters')
    parser.add_argument('--noise', type=float, default=0, help='Optional noise')

    args = parser.parse_args()
    logging.info('Starting clustering with k=%r and noise %r', args.num_clusters, args.noise)
    cluster_vectors(args.input, args.output, args.num_clusters, args.noise)
    logging.info('Done')
