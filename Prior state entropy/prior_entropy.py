import numpy as np
from scipy import stats
import os
import pandas as pd
#from pandas.core.computation.check import NUMEXPR_INSTALLED
import time
import sys
import argparse
import pickle as pkl



def get_transition_matrices(participant): 
    
    
    path = '../Python/Output/4e9c1cf/'

    train_blocks = ['11_20', '186_195']
    chains = ['saw', 'scissors', 'camino', 'silkroad']
    transition_matrices = {'participant': [], 'blocks': [], 'chain':[], 'Pi_matrices':[]}
    for blocks in train_blocks: 
        for chain in chains:
            file_name = '4e9c1cf_{}_elarasztas_{}_blocks_{}_samples.pkl'.format(chain, str(participant), blocks)
            try:
                with open(path + file_name, "rb") as pickle:
                    sample = pkl.load(pickle)
                    Pi_matrices = [item['Pi'] for item in sample['samples']]
                transition_matrices['participant'].append(participant)
                transition_matrices['blocks'].append(blocks)
                transition_matrices['chain'].append(chain)
                transition_matrices['Pi_matrices'].append(Pi_matrices)
            except FileNotFoundError: 
                print ('FileNotFoundError occured')
    return transition_matrices


## NB! Scale up the steps!!! 
def generate_markov_chain_return_entropy(transition_matrix, nr_steps = 1_000_000):
    markov_chain = []
    state = np.random.choice(transition_matrix.shape[0])
    for t in range(nr_steps):
        
        p = transition_matrix[:, :transition_matrix.shape[0]][state]                
        p /= p.sum()
        
        state = np.random.choice(np.arange(transition_matrix.shape[0]), p = p)
        markov_chain.append(state)

    stationary_distr = np.array([markov_chain.count(item) for item in range(transition_matrix.shape[0])])/len(markov_chain)
    entropy  = stats.entropy(stationary_distr)
    #print ('Entropy', np.round(entropy, decimals = 3))
    return (entropy)




def iterate(matrices):

    '''
    This is a function which iterates over each tranistion matrix (800/1600) that is given in a row of 
    'transition_matrices_df.Pi_matrices', and calculate the entropy for the sumulated chains of the matrices. 
    '''

    ### NB!! Here sliceing the matrices[-30:] lets evaluate only the last 30 matrices. 
    ## return [generate_markov_chain_return_entropy(m) for m in matrices[-30:]]
    return [generate_markov_chain_return_entropy(m) for m in matrices[-60:]]



def main():
    start = time.time()
    participant = sys.argv[1]
    ## Here I'm just calling the previously defined functions. 
    transition_matrices = get_transition_matrices(participant)
    transition_matrices_df = pd.DataFrame(transition_matrices)
    transition_matrices_df['entropies'] = transition_matrices_df.Pi_matrices.apply(iterate)
    transition_matrices_df.to_csv(os.getcwd() + '/prior_entropies/' + str(participant) + '_D1_D8_prior_entropy.csv')
    end = time.time()
    elapsed_time = end - start
    print ('prior entropy saved:', participant, 'elapsed time', np.round(elapsed_time, decimals = 1))
    

if __name__ == "__main__":
    main()