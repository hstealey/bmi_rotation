# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:31:49 2023

@author: hanna
"""

"""


'This script contains 6 functions that are used in generatePickles.py'

'_____________________________________________________________________________'
'_____________________________________________________________________________'

[FUNCTION 1] getBehavior_TimeDist

    inputs
        path: full filepath to date folder of interest
        returnCursor: returns the 

    returns
        if returnCursor == False:
            return(dSC, dTime, dDist, dTN, dTargs)
        else:
            print(np.shape(dTargs))
            return(pX, pY, dTN, dTargs)
        
       dSC: dictionary of spike counts aligned to 10Hz cursor updates
       dTime: dictionary of trial times
       dDist: dictionary of trial distances (cursor path length)
       dTN: dictionary of trial numbers for each 10Hz update of dSC
       dTargs: dictionary of target locations for each 10Hz update of dSC
'_____________________________________________________________________________'
'_____________________________________________________________________________'  

[FUNCTION 2]  zeroUnits

    PURPOSE: identifies units that don't have spiking activity within an 8-trial period

    inputs
        dTN: dictionary of trial number for each 10Hz update of dSC
        dTargs: dictionary of target locations for each 10Hz update of dSC
        dSC: dictionary of spike counts aligned to 10Hz cursor updates
        
    return
    
        datesZERO: list of dates (sessions) that correspond to each entry in unitsZERO
        unitsZERO: list of units that have no spiking activity within an 8-trial period
'_____________________________________________________________________________'
'_____________________________________________________________________________'

[FUNCTION 3] compute_scores

    REF: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

    PURPOSE: to compute the negative log-likelihood scores for factor analysis models up to n number of factors (set to 10-Fold CV)
    
    inputs
        X: 2D numpy array that contains the data for the factor analysis model (rows: z-scored (by unit) spike counts x columns: decoder units)
        n_components: list of number of components (factors) to use to fit each model

    returns
        fa_scores: 2D list of scores for each model build on n factors for each fold
        
'_____________________________________________________________________________'
'_____________________________________________________________________________'

[FUNCTION 4] varianceSharedPrivate_byTrial_ALT

    PURPOSE: to decompose neural population variance into shared and private components
        
    inputs
        SC: 2D numpy array that contains the data for the factor analysis model (rows: z-scored (by unit) spike counts x columns: decoder units)
        nf: number of factors used to fit the single factor analysis model
        
    returns
        total: total amount of variance explained
        shared: total amount of shared variance explained
        private: total amount of private variance explained
        loadings: 2D factor loading matrix (can be used to calculate shared variance)
      
'_____________________________________________________________________________'
'_____________________________________________________________________________'

[FUNCTION 5] number_of_factors
    PURPOSE: to determine the number of factors used to fit each FA model
    
    Criteria:
    [1] 'cv': 10-fold cross-validation
    [2] 'sv': 90% of shared variance
    
    inputs:
        dfZERO: see definition above (zeroUnits)
        dates: list of dates (sessions) analyzed
        tBL_list: list of trial times for baseline
        tPE_list: list of trial times for perturbation
        TN: dictionary of trial number
        TL: dictionary of target location
        DX: dictionary of cursor path lengths
        dSC: dictionary of aligned spike counts
        targets: see TL
        subject: string of subject code (e.g., 'airp')
        mode: string of criteria mode ('cv' or 'sv')
        
    
    returns
        dates: list of dates (sessions) analyzed
        numFactors: dictionary of either cross-validation log-likelihood scores (cv mode) or the amount of shared variance explained by each factor model (sv mode)
        numUnits: dictionary containing the number of decoder units for each date, block, trial set number (TSN)
        numBins: number of samples per decoder unit used for FA model fitting for each date, block, trial set number (TSN)
          


'_____________________________________________________________________________'
'_____________________________________________________________________________'

[FUNCTION 6] factor_analysis_final_model

    Purpose: to fit a factor analysis model using the number of factors as determined by the criteria listed in Function 5's definition

    inputs:
        numFactors: dictionary of number of factors to use to fit the FA model (from both modes of Function 5; see generatePickles.py for how the two criteria are resolved)

    returns:
        deg_list: list of rotation applied (degrees) for each session analyzed
        dates: dates of sessions analyzed
        dEV: dictionary of explained variance from factor analysis output (total, shared, private, factor loadings)
        dTimes_, dDist, numUnits, numBins: see Functions 1,2 for definition
        sc_preZ: dictionary of spike counts prior to z-scoring (same as dSC)
        


"""



import os
import glob
import tqdm
import tables
import pickle

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.decomposition import FactorAnalysis

def getBehavior_TimeDist(path, returnCursor=False):
    
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
    pX = {}
    pY = {}
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
        pX[k]         = []
        pY[k]         = []
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
                    pX[k].append(posX[i])
                    pY[k].append(posY[i])
                    # vX[k].append(velX[i])
                    # vY[k].append(velY[i])
                    # trial_num[k].append(j)
           # count+=1
    
        
        print(k, len(np.unique(dTN[k])))
        dSC[k] = np.array(dSC[k]).reshape(-1, num_neurons) 
        # dSC_LIST[k].append(dSC[k])
        # vX_LIST[k].append(vX[k])
        # vY_LIST[k].append(vY[k])


   
       

    if returnCursor == False:
        return(dSC, dTime, dDist, dTN, dTargs)
    else:
        print(np.shape(dTargs))
        return(pX, pY, dTN, dTargs)







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


'############################################################################################'
'Compute 10-Fold Cross-Validation Scores (negative log-likelihood) for Factor Analysis Models'
'############################################################################################'

def compute_scores(X, n_components):
    

    fa = FactorAnalysis(max_iter=10000)
    
    fa_scores = []
    for n in n_components:
    
        fa.n_components = n
        fa_scores.append(cross_val_score(fa, X, cv=10, n_jobs=-1))#fa_scores.append(np.mean(cross_val_score(fa, X, cv=10, n_jobs=-1)))#np.shape(X)[0])))

    return(fa_scores)

'###########################################################'
'Factor Analysis by Trial Set Number (TSN; sets of 8 trials)'
'###########################################################'


def varianceSharedPrivate_byTrial(SC, nf):

    '''
    This function uses factor analysis to parse the total population variance into
    shared and private components. These calculations are performed separately 
    for sets of 8 trials.
    '''
    

    FA = FactorAnalysis(n_components=nf, max_iter=10000)
    FA.fit(SC)
    
    shared_variance = FA.components_.T.dot(FA.components_)
    private_variance = np.diag(FA.noise_variance_)

    
    total   = np.trace(shared_variance) + np.trace(private_variance) #equivalent: total = np.trace(shared_variance+private_variance)
    shared  = np.trace(shared_variance) 
    private = np.trace(private_variance)                
    
    weights = FA.components_
    loadings = weights
    
    return(total, shared, private, loadings) 

'##########################################################################'
' Determining the Number of Factors CROSS-VALIDATION, 90% SHARED VARIANCE '
'##########################################################################'


def number_of_factors(dfZERO, dates, tBL_list, tPE_list, TN, TL, DX, dSC, targets, subject, mode):
    
    dTimes_ = {}
    dDist_  = {}

    
    numUnits = {}
    numBins  = {}
    numFactors = {}
    
    
    for d, date in enumerate(tqdm.tqdm(dates)):#[38,41,48]:
       
        dTimes_[d] = {}
        dDist_[d]  = {}
        
 
        numUnits[d]   = {}
        numBins[d]    = {}
        numFactors[d] = {}
  
        
    
        for k in ['BL', 'PE']:
            
            numUnits[d][k]   = []
            numBins[d][k]    = []
            
            if mode == 'cv':
                'Cross-Validation'
                numFactors[d][k] = []
            elif mode == 'sv':
                'Shared Variance'
                numFactors[d][k] = {'total': [], 'shared': [], 'private':[], 'loadings':[], 'shared90':[]}
                

         
            if k == 'BL':
                times = tBL_list[d]
            elif k == 'PE':
                times = tPE_list[d]
            
            
            'Trials and their Target Locations'
            trials = TN[d][k]
            targs  = np.array(targets[d][k])
            dfUpdates = pd.DataFrame({'trials': trials, 'targs':targs})
        
            
            'Determine the indices of the spike count rows that below to each trial (and corresponding target location).'
            inds = []
            targ_inds = []
            for t in np.unique(dfUpdates['trials']):
                temp = dfUpdates.loc[dfUpdates['trials']==t].index.tolist()
                inds.append(temp)
                targ_inds.append(dfUpdates['targs'][temp[0]])
           
            'Determine the minimum number of trials for a given target location.'
            targetDeg = np.arange(0,360,45)
            minTrials = [len(np.where(np.array(targ_inds) == deg)[0])  for deg in targetDeg]
            minNumTEST = min(minTrials)
            
            if minNumTEST < 40:
                print(d, 'FEWER THAN 40 TRIALS')
            
            minNum = 40
            'Determine the indices of each trial type'
            TargInds   = np.zeros((8, minNum))
            TrialTimes = np.zeros((8, minNum))
            TrialDists = np.zeros((8, minNum))
            
            for i, deg in zip(np.arange(8), targetDeg):
                temp = np.where(np.array(targ_inds) == deg)[0]
                TargInds[i,:] = temp[:minNum]
                
                for ti in range(minNum):
                    TrialTimes[i,ti] = times[temp[ti]]
                    TrialDists[i,ti] = DX[d][k][temp[ti]]
                
            dTimes_[d][k] = TrialTimes
            dDist_[d][k]  = TrialDists
            
            'Perform FA on a set of 8 trials.'
            
            
            for j in range(minNum): 
                tempSV = []
             
                set_targ_inds = (TargInds[:,j]).astype(int).tolist()
                temp = np.concatenate([inds[i] for i in set_targ_inds]).tolist()
                sc = np.array(dSC[d][k])[temp,:]
    
                units_to_drop = dfZERO.loc[dfZERO['d']==date, 'unit'].tolist()
                sc_fixed = np.delete(sc, units_to_drop, axis=1)
    
                
                '_________________________________________________________________'
                
                'Factor Analsysis'
                
                """
                    k-fold cross-validation
                
                    OR
                
                    90% variance
                    
                """            
                
                '_________________________________________________________________'
                
                if np.any(np.isnan(np.sum(stats.zscore(sc_fixed, axis=0), axis=0))) == 1:
                    temp_zero = np.where(np.sum(sc_fixed, axis=0) == 0)[0][0]
                    sc_fixed = np.delete(sc_fixed, temp_zero, axis=1)
                    print('ZERO ERROR')
                
                X = stats.zscore(sc_fixed, axis=0)
                X = X[:,:]
                
                n_bins  = np.shape(X)[0]
                n_units = np.shape(X)[1]
                
                numBins[d][k].append(n_bins)
                numUnits[d][k].append(n_units)
                
                if mode == 'cv':
                    '10-Fold Cross Validation'
                    if subject == 'airp':
                        n_components = np.arange(n_units-10)
                    elif subject == 'braz':
                        n_components = np.arange(50)
                    else:
                        print('ERROR in factorAnalysis_fxn subject')
        
                    fa_scores = compute_scores(X, n_components)
                    nf = np.where(fa_scores == np.max(fa_scores))[0][0]
                    
                    numFactors[d][k].append(fa_scores)
                
                elif mode == 'sv':
                    'Determining 90% of Shared Variance'
                    nf = n_units
                    total, shared, private, loadings = varianceSharedPrivate_byTrial(X, nf)
                    sv90 = shared*0.9
                    
                    
                    total_ = []
                    shared_ = []
                    private_ = []
                    loadings_ = []
                    
                    for numF in range(n_units): #TODO: Full number?
                        total, shared, private, loadings = varianceSharedPrivate_byTrial(X, numF)
                        
                        total_.append(total)
                        shared_.append(shared)
                        private_.append(private)
                        loadings_.append(loadings)
                        
                    numFactors[d][k]['total'].append(total_)
                    numFactors[d][k]['shared'].append(shared_)
                    numFactors[d][k]['private'].append(private_)
                    numFactors[d][k]['loadings'].append(loadings_)
                
                    
                print(date, k, j)
            print(date)
            
    return(dates, numFactors, numUnits, numBins)


def factor_analysis_final_model(numFactors, dfZERO, dates, tBL_list, tPE_list, TN, TL, DX, dSC, targets, subject):
        
    dTimes_ = {}
    dDist_  = {}

    
    numUnits = {}
    numBins  = {}
    dEV      = {}
    sc_preZ  = {}
    
    
    for d, date in enumerate(tqdm.tqdm(dates)):#[38,41,48]:
       
        dTimes_[d] = {}
        dDist_[d]  = {}
        
 
        numUnits[d]   = {}
        numBins[d]    = {}
        dEV[d]        = {}
        sc_preZ[d]    = {}
  
    
        for k in ['BL', 'PE']:
            
            numUnits[d][k]   = []
            numBins[d][k]    = []
            dEV[d][k]        = {'total': [], 'shared': [], 'private':[], 'loadings':[]}
            sc_preZ[d][k]    = []
            
            if k == 'BL':
                times = tBL_list[d]
            elif k == 'PE':
                times = tPE_list[d]
            
            
            'Trials and their Target Locations'
            trials = TN[d][k]
            targs  = np.array(targets[d][k])
            dfUpdates = pd.DataFrame({'trials': trials, 'targs':targs})
        
            
            'Determine the indices of the spike count rows that below to each trial (and corresponding target location).'
            inds = []
            targ_inds = []
            for t in np.unique(dfUpdates['trials']):
                temp = dfUpdates.loc[dfUpdates['trials']==t].index.tolist()
                inds.append(temp)
                targ_inds.append(dfUpdates['targs'][temp[0]])
           
            'Determine the minimum number of trials for a given target location.'
            targetDeg = np.arange(0,360,45)
            minTrials = [len(np.where(np.array(targ_inds) == deg)[0])  for deg in targetDeg]
            minNumTEST = min(minTrials)
            
            if minNumTEST < 40:
                print(d, 'FEWER THAN 40 TRIALS')
            
            minNum = 40
            'Determine the indices of each trial type'
            TargInds   = np.zeros((8, minNum))
            TrialTimes = np.zeros((8, minNum))
            TrialDists = np.zeros((8, minNum))
            
            for i, deg in zip(np.arange(8), targetDeg):
                temp = np.where(np.array(targ_inds) == deg)[0]
                TargInds[i,:] = temp[:minNum]
                
                for ti in range(minNum):
                    TrialTimes[i,ti] = times[temp[ti]]
                    TrialDists[i,ti] = DX[d][k][temp[ti]]
                
            dTimes_[d][k] = TrialTimes
            dDist_[d][k]  = TrialDists
            
            'Perform FA on a set of 8 trials.'
            
            
            for j in range(minNum): 
                tempSV = []
             
                set_targ_inds = (TargInds[:,j]).astype(int).tolist()
                temp = np.concatenate([inds[i] for i in set_targ_inds]).tolist()
                sc = np.array(dSC[d][k])[temp,:]
    
                units_to_drop = dfZERO.loc[dfZERO['d']==date, 'unit'].tolist()
                sc_fixed = np.delete(sc, units_to_drop, axis=1)
    
                
                '_________________________________________________________________'
                
                'Factor Analsysis'
                
                """
                    k-fold cross-validation
                
                    OR
                
                    90% variance
                    
                """            
                
                '_________________________________________________________________'
                
                if np.any(np.isnan(np.sum(stats.zscore(sc_fixed, axis=0), axis=0))) == 1:
                    temp_zero = np.where(np.sum(sc_fixed, axis=0) == 0)[0][0]
                    sc_fixed = np.delete(sc_fixed, temp_zero, axis=1)
                    print('ZERO ERROR')
                    
                sc_preZ[d][k].append(sc_fixed)   
                
                X = stats.zscore(sc_fixed, axis=0)
                X = X[:,:]
                
                n_bins  = np.shape(X)[0]
                n_units = np.shape(X)[1]
                
                numBins[d][k].append(n_bins)
                numUnits[d][k].append(n_units)
                
                nf = int(numFactors[d][k][j])
                total, shared, private, loadings = varianceSharedPrivate_byTrial(X, nf)
                    
    
                dEV[d][k]['total'].append(total)
                dEV[d][k]['shared'].append(shared)
                dEV[d][k]['private'].append(private)
                dEV[d][k]['loadings'].append(loadings)
            
                    
                print(date, k, j)
            print(date)
            
    return(deg_list, dates, dEV, dTimes_, dDist, numUnits, numBins, sc_preZ)  
