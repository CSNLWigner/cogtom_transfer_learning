import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None



def import_and_transform_data(path, csv_name):
    data = pd.read_csv(path + csv_name, index_col=[0])
    data = data.loc[data['trial_type'] != 'Prac']
    data = data.loc[data['correct_response'] == 1]
    data.reset_index(inplace = True)
    data = data.rename(columns = {'participant_test':'participant'})
    data.loc[:, 'blocks_from'] = data.e_train.str.split('_').map(lambda x: x[0])
    data.loc[:, 'blocks_to'] = data.e_train.str.split('_').map(lambda x: x[1])
    data.blocks_from = data.blocks_from.astype('int')
    data.blocks_to = data.blocks_to.astype('int')

    days = {1:(1,25), 2:(26,50), 3:(51,75), 4:(76,100), 5:(101, 125),
            6:(126,150), 7:(151,175), 8:(176,200), 9:(201,225), 10:(226,245)}


    for day in days.keys():
        sessions_from, sessions_to = days[day][0], days[day][1]
        for r in range(sessions_from, sessions_to+1):
            data.loc[data['blocks_from'] == r, 'day'] = day

    ################
    # Get data tag
    ################
    n_datapoints = data.shape[0]
    chains = data.ini.unique()
    e_train = data['e_train'].unique()
    e_test = data['e_test'].unique()
    data_tag = 'Nr of datapoints: {:,} \nchains:{} \ntrain:{}, test:{} \ndata used: {}'.format(n_datapoints, chains, e_train, e_test, csv_name)

    return (data_tag, data)






def drop_RTs_above_3std_below_180(data):
    rt_means = dict(data.groupby('participant')['rt'].mean())
    participants = rt_means.keys()
    rt_means = np.array(data.groupby('participant')['rt'].mean())
    rt_std = np.array(data.groupby('participant')['rt'].std())
    rt_mean_3std = rt_means + 3*rt_std
    rt_mean_3std = dict(zip(participants, rt_mean_3std))
    data['rt_means_3std'] = data['participant'].map(rt_mean_3std)
    data = data.loc[data['rt'] < data['rt_means_3std']]

    ## drop RTs below 180ms and above 5000 ms
    data = data.loc[data['rt']> 180]
    data = data.loc[data['rt']< 5000]
    return (data)






def get_corr_mtrx(data):
    corr_matrix = (data
                       .groupby(['model', 'participant', 'e_train', 'e_test', 'day', 'block'])
                       [['rt','rt_predicted']]
                       .corr())
    corr_matrix.reset_index(inplace = True)
    corr_matrix.drop(['level_6', 'rt'], axis = 1, inplace = True)
    matrix = corr_matrix.loc[corr_matrix['rt_predicted'] !=1]
    matrix['r_sqr'] = matrix.rt_predicted**2
    matrix.drop('rt_predicted', 1, inplace = True)
    return matrix



def get_ct_m_diff(matrix):
    data_ct_m =  matrix.pivot(index = ['participant', 'block'], columns = 'model', values = 'r_sqr')
    data_ct_m['m_ct_diff'] = data_ct_m['iHMM'] - data_ct_m['Markov'] 
    return data_ct_m

def get_data_by_blocks(data_ct_m):
    data_block = data_ct_m.reset_index()
    data_block = pd.DataFrame(data_block)
    data_block = (data_block
                        .groupby(['participant', 'block'])
                        .mean()        
                 )
    data_block = data_block[['iHMM', 'Markov', 'm_ct_diff']].stack().reset_index()
    data_block = data_block.pivot(index =['model', 'block'], columns = 'participant')
    data_block.columns = data_block.columns.droplevel(0)
    return data_block