Within this analysis, we simulated Markov chains of 1m steps based on the transition matrices captured by CT on D1 and D8 for each participant.

## Simulation and analysis steps: 
1. Generate Markov chains and calculate entropy
2. Read and analyze the prior entropies




#### 1. Generate Markov chains and calculate entropy
prior_entropy.py

Run prior entropy file: 

from the Python_analysis folder: 
	python prior_entropy.py 109
	to avoid unexpected logouts: nohup python prior_entropy.py 109  > log.txt 2>&1 &


Notes: 
It might be useful to reduce the number of steps (It should be validated, whether 10_000 steps produce the same result as 1_000_000 steps with respect to the entropy).
It would have been useful to write into the name that how many iterations used during the 


#### 2. Read and analyze the prior entropies
(To do!!) Prior entropy of the hidden states - ANALYSIS.ipynb
