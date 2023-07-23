# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:31:49 2023

@author: hanna
"""


import os
import glob
import tables
import pickle

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.decomposition import FactorAnalysis

def getBehavior_TimeDist(path):
    
    os.chdir(path)
    
    hdf = tables.open_file(glob.glob('*.hdf')[0])
    decoder_files = glob.glob('*.pkl')
    KG_picklename = decoder_files[1]
    
    
    '''Load KG values for task.'''
    f = open(KG_picklename, 'rb')
    KG_stored = []
    while True:
        try:
            KG_stored.append(pickle.load(f))
        except Exception:
            break
    f.close()
    
    
    '''Parse HDF file.'''
    spikes = hdf.root.task[:]['spike_counts'] # (inds, neurons, 1)
    num_neurons = np.shape(spikes)[1]
    
    msg = hdf.root.task_msgs[:]['msg'] 
    ind = hdf.root.task_msgs[:]['time']
    
    deg = hdf.root.task._v_attrs['rot_angle_deg']
    
    pert = hdf.root.task[:]['pert'][:,0]
    error_clamp = hdf.root.task[:]['error_clamp'][:,0]
    
    reward_msg = np.where(msg == b'reward')[0]
    holdCenter_msg = np.subtract(reward_msg,5)
    
    reward_ind = ind[reward_msg]
    holdCenter_ind = ind[holdCenter_msg]
    
    block_type = hdf.root.task[:]['block_type'][:,0]
    
    df = pd.DataFrame({'hold': holdCenter_ind, 
                       'reward': reward_ind, 
                       'pert': pert[reward_ind], 
                       'errorClamp': error_clamp[reward_ind], 
                       'blockType': block_type[reward_ind]})
    
    trialsStart = [np.where(df['blockType'] == i)[0][0]  for i in np.unique(block_type)]
    trialsEnd   = [np.where(df['blockType'] == i)[0][-1] for i in np.unique(block_type)]
    
    if len(np.unique(block_type)) == 3:
        keys = ['BL', 'PE', 'PL']
        numB = 3
    elif len(np.unique(block_type)) == 4:
        keys = ['BL', 'PE', 'PL', 'WO']
        numB = 4
    else:
        print("ERROR IN NUMBER OF BLOCKS")

    decoder_state = hdf.root.task[:]['decoder_state']
    
    cursor = hdf.root.task[:]['cursor']
    x = cursor[:,0]
    y = cursor[:,2]
    
    cursorx = x
    cursory = y
        
        
    '###################################'
    ''' Determining Target Locations '''
    '###################################'
    
    dfTarget = pd.DataFrame(hdf.root.task[:]['target'])
    
    dfTarget['degrees'] = np.arctan2(dfTarget[2], dfTarget[0]) * (180/np.pi) 
    dfTarget['degrees'] = list(map(lambda d:d+360 if d < 0 else d, dfTarget['degrees']))
    
    #Trim to just the reward indices
    dfTargetLoc = pd.DataFrame(columns=['degrees', 'holdCenter_ind', 'reward_ind'])
    dfTargetLoc['degrees'] = dfTarget['degrees'][reward_ind]
    dfTargetLoc['holdCenter_ind'] = holdCenter_ind
    dfTargetLoc['reward_ind'] = reward_ind
    dfTargetLoc = dfTargetLoc.reset_index(drop=True)
    
    targetDeg = np.unique(dfTargetLoc['degrees'])
    
    df['target'] = dfTarget['degrees'][df['reward']].tolist()
    
    '############################'
    ''' Decoder update inds based on cursor values and checked against how many times KG was stored. '''
    '############################'
    
    lenHDF = len(hdf.root.task)
    lenKG  = len(KG_stored)
    
    posX = decoder_state[:,0,0]
    posY = decoder_state[:,2,0]
    
    velX = decoder_state[:,3,0]
    velY = decoder_state[:,5,0]
    
    first_update = np.min([np.where(posX > 0)[0][0], np.where(posY > 0)[0][0], np.where(velX > 0)[0][0], np.where(velY > 0)[0][0]])
    
    
    count = 0
    update_inds = []
    for i in range(lenHDF):
        if count < 5:
            update_inds.append(0)
            count+=1
        elif count == 5:
            update_inds.append(1)
            count = 0
    

    dTrials = {} #Without error clamp trials
    dTime  = {}
    
    for j,k in enumerate(keys):
    
        nonEC_trials = []
        times = []
        
        for i in range(trialsStart[j], trialsEnd[j]):
            if df['errorClamp'][i] == 0:
                s = df['hold'][i]
                e = df['reward'][i]
                nonEC_trials.append(i)
                times.append( (e-s)/ 60)
        dTrials[k] = nonEC_trials
        dTime[k]  = times
        
    'Total Distance Cursor Travels (i.e., path length)'
    dDist = {}
    
    for k in keys:
        
        dDist[k] = []
        
        trials     = dTrials[k]
        targs      = dfTargetLoc['degrees'][trials].tolist()
        start_inds = dfTargetLoc['holdCenter_ind'][trials]
        end_inds   = dfTargetLoc['reward_ind'][trials]
    
        
        count=0
        for i,j in zip(start_inds, end_inds):


            x_ = [x[i]]
            y_ = [y[i]]
            
            for q in range(i+1,j):
                if (x[q] != x_[-1]) or (y[q] != y_[-1]):
                    x_.append(x[q])
                    y_.append(y[q])
            
            dx = np.diff(x_)
            dy = np.diff(y_)
            
            dDist[k].append(np.sum(np.sqrt(dx**2 + dy**2)))

            
            count+=1
            
    '################################################'    
    ''' Aligned spike counts [dictionaries]'''
    '################################################' 
    
    dSC = {}
    # pX = {}
    # pY = {}
    # vX = {}
    # vY = {}
    dTN = {} #trial_num = {}
    dTargs = {}
    
    dBN = {1:'BL', 2:'PE', 3:'PL', 4:'WO'}
    
    #count=0
    for blockNum in range(1,numB+1):
        k = dBN[blockNum]
        
        dSC[k]    = []
        dTN[k]    = []
        dTargs[k] = []
        # pX[k]         = []
        # pY[k]         = []
        # vX[k]         = []
        # vY[k]         = []
        # trial_num[k]  = []
        
    
        inds = df.loc[ (df['blockType'] == blockNum) & (df['errorClamp'] == 0)].index.tolist()
     
        #print(df.keys())   
     
        for j in inds:
            s = df['hold'][j]
            e = df['reward'][j]
            targ = df['target'][j]
            
            for i in range(s,e):
                if update_inds[i] == 1:
                    dSC[k].append(np.sum(spikes[i-5:i+1, :, 0], axis=0))
                    dTN[k].append(j) 
                    dTargs[k].append(targ)
                    # pX[k].append(posX[i])
                    # pY[k].append(posY[i])
                    # vX[k].append(velX[i])
                    # vY[k].append(velY[i])
                    # trial_num[k].append(j)
           # count+=1
    
        
        print(k, len(np.unique(dTN[k])))
        dSC[k] = np.array(dSC[k]).reshape(-1, num_neurons) 
        # dSC_LIST[k].append(dSC[k])
        # vX_LIST[k].append(vX[k])
        # vY_LIST[k].append(vY[k])


   
       


    return(dSC, dTime, dDist, dTN, dTargs)







'#############################'
'      UNITS TO DELETE        '
'#############################'



def zeroUnits(dTN, dTargs, dSC, date):

    datesZERO = []
    unitsZERO = []
    
    for k in ['BL', 'PE']:
        
        kSC = np.array(dSC[k])
        
        'Trials and their Target Locations'
        trials = dTN[k]#.tolist()
        targs  = np.array(dTargs[k])
        dfUpdates = pd.DataFrame({'trials': trials, 'targs':targs})
        
        
        'Determine the indices of the spike count rows that below to each trial (and corresponding target location).'
        inds = []
        targ_inds = []
        for t in np.unique(dfUpdates['trials']):
            temp = dfUpdates.loc[dfUpdates['trials']==t].index.tolist()
            inds.append(temp)
            targ_inds.append(dfUpdates['targs'][temp[0]])
       
        TargInds = np.zeros((8,40))
        minNum = 40
        targetDeg = np.arange(0,360,45)
        for i, deg in enumerate(targetDeg):
            temp = np.where(np.array(targ_inds) == deg)[0]
            TargInds[i,:] = temp[:minNum]
            
        
        'Test if Unit Has 0 Spikes for Any Given Set (used in FA)'
        for j in range(minNum): 
            set_targ_inds = (TargInds[:,j]).astype(int).tolist()
            temp = np.concatenate([inds[i] for i in set_targ_inds]).tolist()
            sc = kSC[temp,:]
           
            temp2 = np.sum(sc, axis=0)
            temp22 = np.sum(np.isnan(stats.zscore(sc, axis=0)), axis=0)
    
            if len(np.where(temp2 == 0)[0]) != 0:
                datesZERO.append(date)
                unitsZERO.append(np.where(np.array(temp2) == 0)[0][0])
                #ZERO_UNITS.append([d, np.where(np.array(temp2) == 0)[0][0]])
                
            if len(np.where(temp22 >=1)[0]) !=0:
                temp222 = list(np.where(temp22 >=1)[0])
                for item in temp222:
                    #ZERO_UNITS.append([d, item])
                    datesZERO.append(date)
                    unitsZERO.append(item)
                
 
    print(date)
    return(datesZERO, unitsZERO)


def varianceSharedPrivate_byTrial(SC):

    '''
    This function uses factor analysis to parse the total population variance into
    shared and private components. These calculations are performed separately 
    for sets of 8 trials.
    '''
    

    numUnits = np.shape(SC)[1]
    
    FA = FactorAnalysis(n_components=numUnits)
    FA.fit(SC)
    
    shared_variance = FA.components_.T.dot(FA.components_)
    private_variance = np.diag(FA.noise_variance_)
    
    tot_shared  = [shared_variance[i,i] for i in range(numUnits)]
    tot_private = [private_variance[i,i] for i in range(numUnits)]

    weights = FA.components_
    
    return(tot_shared, tot_private, weights)
