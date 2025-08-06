import pickle
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def read_pkl_file_chain_pocoMC(PATH_pklfile_chain):
    """
    Reads a pickle file containing the chain data from pocoMC.
    The expected structure of the data is:
    - 'chain'
    - 'weights'
    - 'logl'
    - 'logp'
    - 'logz'
    - 'logz_err'
    This function returns the data as a dictionary.
    """
    with open(PATH_pklfile_chain, 'rb') as pf:
        data = pickle.load(pf)

    return data

def sort_chain_likelihood(PATH_pklfile_chain):
    """
    Sorts the chain data based on the log likelihood values in descending order.
    The sorted data is saved to a new pickle file with '_sorted' appended to the original filename.
    """
    run_chain = read_pkl_file_chain_pocoMC(PATH_pklfile_chain)
    array_chain = run_chain['chain']
    array_weights = run_chain['weights']
    array_logl = run_chain['logl']
    array_logp = run_chain['logp']
    array_logz = run_chain['logz']
    array_logz_err = run_chain['logz_err']

    # sort the array_logl with decreasing order and get the indices
    sorted_indices = np.argsort(array_logl)[::-1]

    sorted_array_logl = array_logl[sorted_indices]
    # sort also the other arrays
    sorted_array_chain = array_chain[sorted_indices]
    sorted_array_weights = array_weights[sorted_indices]
    sorted_array_logp = array_logp[sorted_indices]

    # write a new file with the same content and the sorted data
    data = {'chain': sorted_array_chain, 
            'weights': sorted_array_weights, 
            'logl': sorted_array_logl, 
            'logp': sorted_array_logp, 
            'logz': array_logz, 
            'logz_err': array_logz_err
            }
    
    with open(PATH_pklfile_chain.replace('.pkl', '_sorted.pkl'), 'wb') as f:
        pickle.dump(data, f)

def generate_posterior_clusters(PATH_pklfile_chain_sorted, num_samples=None, num_clusters=10):
    """
    Generate posterior clusters from the sorted chain file.
    """
    run_chain = read_pkl_file_chain_pocoMC(PATH_pklfile_chain_sorted)
    array_chain = run_chain['chain']
    if num_samples is not None:
        array_chain = array_chain[:num_samples]

    scaler = StandardScaler()
    scaled_chain = scaler.fit_transform(array_chain)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_chain)

    kmeans_clustered_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Save the cluster centers to a txt file
    np.savetxt("cluster_centers.txt", kmeans_clustered_centers.T, fmt='%.6f')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python sort_chain_likelihood.py <path_to_chain_file> <number_of_most_likely_samples_considered> <number_of_clusters>")
        print("Arguments:")
        print("  <path_to_chain_file>: Path to the pickle file containing the chain data from pocoMC.")
        print("  <number_of_most_likely_samples_considered>: Number of most likely samples to consider for clustering. Use 'None' to consider all samples.")
        print("  <number_of_clusters>: Number of clusters to generate.")
        sys.exit(1)
    PATH_pklfile_chain = sys.argv[1]
    num_samples_str = sys.argv[2]
    num_samples = None if num_samples_str == 'None' else int(num_samples_str)
    num_clusters = int(sys.argv[3])
    sort_chain_likelihood(PATH_pklfile_chain)
    PATH_pklfile_chain_sorted = PATH_pklfile_chain.replace('.pkl', '_sorted.pkl')
    generate_posterior_clusters(PATH_pklfile_chain_sorted, num_samples, num_clusters)
    print(f"Posterior clusters generated and saved to 'cluster_centers.txt'.")