import pandas as pd
import numpy as np
import itertools
from itertools import chain



def get_stars(p_val):
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = ''
    return stars




def generate_all_sequences():
    all_sequences = list(itertools.permutations([0, 1, 2, 3]))
    all_sequences_s = [[''.join(str(c) for c in s)] for s in all_sequences]
    all_sequences_s = list(chain(*all_sequences_s))
    all_sequences_s = [item*2 for item in all_sequences_s]


    # print(len(all_sequences))
    # print(all_sequences_s)


    individual_sequences = []

    for s in all_sequences_s:
        if len (individual_sequences) == 0:
            individual_sequences.append(s)
            #print (s)
        else: 
            if len([item for item  in individual_sequences if s[:4] in item]) > 0:
                pass
                #print (s, 'has an overlap with', [item[:4] for item  in individual_sequences if s[:4] in item])

            else: 
                #print (s, 'is new')
                individual_sequences.append(s)

    return (individual_sequences)




def generate_complete_seq_tab():
    sequences = pd.read_csv('/Users/szekelyanna/Dropbox/CSNL/cognitive_tomography/analysis_2022/data/sequences.csv',
                        dtype={'176_185': str, '201_210': str}).set_index('participants')

    
    all_sequences =  generate_all_sequences()
    participants = sequences.index.to_list()

    alter_sequences = {}
    for p in participants:
        alter_seqs = [item[:4] for item in all_sequences if item[:4] not in sequences.loc[p][0]*2 and item[:4] not in sequences.loc[p][1]*2]
        alter_sequences[p] = alter_seqs

    alternative_sequences = pd.DataFrame(alter_sequences).transpose()
    alternative_sequences.columns = ['alt1', 'alt2', 'alt3', 'alt4']
    sequences = sequences.join(alternative_sequences)
    sequences.rename(columns = {'176_185': 'D8_seq','201_210': 'D9_seq'}, inplace=True)


    ## here I orginse D8 and D9 sequences to the form of the other sequences, starting with zero.
    ## REally bad idea.....
    """     sequences['176_185'] = sequences['176_185'].astype('string')
    sequences['201_210'] = sequences['201_210'].astype('string')
    D8_seq = sequences['176_185']*2
    D9_seq = sequences['201_210']*2
    sequences['D8_seq'] = '0'+ D8_seq.str.split('0', 1).str[1].str[0:3]  
    sequences['D9_seq'] = '0'+ D9_seq.str.split('0', 1).str[1].str[0:3]
    sequences.drop('176_185', axis = 1, inplace=True)
    sequences.drop('201_210', axis = 1, inplace=True)
    sequences = sequences.astype('string') """

    return (sequences)




def get_permuted_sequence(p, sq_to):
    sequences = generate_complete_seq_tab()
    
    '''
    This method is to generate an alternative column order. 
    It takes the participant and the alternative sequence name (header in the sequences table) and generate 
    new sequence which can be fed to the permute_phi() method. 
    '''
    
    _from = [item for item in sequences.loc[p,:]['D8_seq']] ## get the D8 sequence for participant p
    _to = [item for item in sequences.loc[p,:][sq_to]] ## get the D9 sequence for participant p
    
    print (p, 'seqs: from:', _from, 'to:',_to)
    sequence_map = (dict(zip(_from, _to))) ## create a dict from the sequences: {D8:D9}
    sequence_map = dict(sorted(sequence_map.items())) ## we sort the dictionary keys (D8 seqence) ascending: y0, y1, y2, y3
    permuted_sequence = list(sequence_map.values()) ## keep only the values of the dicttionary (D9 seqence),
                                                    ## the obtained list will give the information how the columns 
                                                    ## of D8 should be ordered. 
            
    permuted_sequence = ['c'+item for item in permuted_sequence]
    return (permuted_sequence)




def get_gen_probs(data):
    gen_probs_data = data.loc[data['model'] == 'Markov'][['Y', 'trial_type', 'participant_test',
                                                          'e_train', 'block', 'ini', 'trial']]
                                                            ## Filtering for markov is 
                                                            ## neccessary just in order 
                                                            ## to avoid duplicates in the df. 

    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] == 0), 'y0'] = 1
    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] != 0), 'y0'] = 0

    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] == 1), 'y1'] = 1
    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] != 1), 'y1'] = 0

    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] == 2), 'y2'] = 1
    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] != 2), 'y2'] = 0

    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] == 3), 'y3'] = 1
    gen_probs_data.loc[(gen_probs_data ['trial_type'] == 'P') & (gen_probs_data['Y'] != 3), 'y3'] = 0

    gen_probs_data.loc[gen_probs_data ['trial_type'] == 'R', 'y0'] = 0.25
    gen_probs_data.loc[gen_probs_data ['trial_type'] == 'R', 'y1'] = 0.25
    gen_probs_data.loc[gen_probs_data ['trial_type'] == 'R', 'y2'] = 0.25
    gen_probs_data.loc[gen_probs_data ['trial_type'] == 'R', 'y3'] = 0.25
    
    return gen_probs_data




def map_days_int(data, new_col_name = 'day', col_to_map_on = 'e_train'):

    days_int = {'11_20':1,
                '36_45':2,
                '61_70':3,
                '86_95':4,
                '111_120':5,
                '136_145':6,
                '161_170':7,
                '186_195':8,
                '211_220':9,
                '236_245':10}

    data[new_col_name] = data['e_train'].map(days_int)
    return datax