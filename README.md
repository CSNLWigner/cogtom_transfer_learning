# cogtom_transfer_learning
This repo contains the codes and analyses for the **COGTOM TRANSFER LEARNING** project. 

Manuscript: <a href="https://github.com/CSNLWigner/cogtom_transfer_learning/tree/main"> Identifying transfer learning in the reshaping of inductive biases. </a>

# REPO CONTENT

## Data

Experimental data: stimuli and measured RTs are in elarasztas_dataset.zip. 

## Scripts for the analyses

#### LEARNING CURVES, AND LEARNING STRENGTH ANALYSES

(Training and transfer pahse)
- Learning Curves D1-D8 (ALL PARTICIPANTS).ipynb
- CT = GT+Markov, LS on D1 vs. D9.ipynb
- CT, Markov, GT on D1, D8.ipynb
- Learning curves (111, 119), LS on transfer vs. LS on training.ipynb
- Learning strength and D8 model performances.ipynb

  

#### ACROSS SESSIONS ANALYSES
(Training and transfer pahse)

- Across sessions - crossover plot (111, 119).ipynb
- Transfer evaluation.ipynb




#### ALTERNATION PHASE ANALYSES
- Alternating model performance.ipynb
- Alternation score vs. learning strength.ipynb




#### PERMUTED INTERNAL MODEL ANALYSES
- Sigma(CT) vs. sigma(GT).ipynb
- Normative vs. alternative permutations.ipynb

Data manipulation: Internal model permutation
- Permuting the Phi matrix.ipynb
- sequences.csv




#### HIDDEN STATE ANALYSES
- Cross-Entropy of the Hidden States.ipynb
- Prior entropy of the hidden states.ipynb

Data simulation: prior_entropy.py

Simulated data: prior_entropy.py (csvs with entropies for each participants)




#### HELPER FUNCTIONS

- data_import.py
- helper.py
- plot_params.py
- curlyBrace.py
