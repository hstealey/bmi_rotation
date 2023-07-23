# -*- coding: utf-8 -*-
"""
Purpose: 
    
    [1] To decompose sources of variability over "time"
        Method: factor analysis (varianceSharedPrivate_byTrial)
        "time": FA is performed over sets of 8 trials; 8 is significant because
                it's the number of target locations; 
                each appears randomly within a set of 8 trials.
    [2] To combine behavioral metrics (distance cursor travels, time of trials) from all sessions
    
    
Last update: April 23, 2023
@author: hanna
"""


import os
import pickle
import numpy as np
from glob import glob
from scipy import stats
import pandas as pd
from bioinfokit.analys import stat

os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')
from generatePickles_fxns import  getBehavior_TimeDist, zeroUnits, varianceSharedPrivate_byTrial #, getdSC

dSubjects  = {0:['airp','Airport'], 1:['braz','Brazos']}
targetDeg = np.arange(0,360,45)

subject_ind = 0
subject = dSubjects[subject_ind][0]
subj = dSubjects[subject_ind][1]
root_path = r'C:\BMI_Rotation\Data'+'\\'+subj


'###################################################################'
'''  Extracting Aligned Spike Counts, Trial Times, Distances      '''
'###################################################################'

try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    open_file = open(subject+'_Behavior_TimeDist.pkl', "rb") 
    loaded = pickle.load(open_file)
    deg_list, date_list, dSC_, dTime, dDist, dTN, dTargs = loaded
    open_file.close()

except:
    
    dSC_   = {}
    dTime = {}
    dDist = {}
    dTN = {}#trial_num = {}
    dTargs = {}#target_loc = {}
    
    date_list = []
    deg_list = []
    
    
    os.chdir(root_path)
    degFolders = glob('*')
    
    for degFolder in degFolders:
        os.chdir(root_path+'\\'+degFolder)
        dates = glob('*')
        
        
        for date in dates:
            dSC_[date], dTime[date], dDist[date], dTN[date], dTargs[date] = getBehavior_TimeDist(root_path+'\\'+degFolder+'\\'+date)
            deg_list.append(degFolder)
            date_list.append(date)
            
        
            print('Finished:', degFolder, date)

    '''Saving values to .pkl'''     
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = [deg_list, date_list, dSC_, dTime, dDist, dTN, dTargs]
    filename = subject+'_Behavior_TimeDist.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()
 
#%%
'#############################################################################'
'''  Determining Decoder Units with "0" Spike Count in any Set of Trials    '''
'#############################################################################'

try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    filename = subject+'_zeroUNITS.pkl' #'_FA_units_failed.pkl'
    open_file = open(filename, "rb")
    loaded = pickle.load(open_file)
    dfZERO = loaded
    open_file.close()

except:

    datesZERO = {}
    unitsZERO = {}
    
    os.chdir(root_path)
    degFolders = glob('*')

    for degFolder in degFolders:
        os.chdir(root_path+'\\'+degFolder)
        dates = glob('*')
        
        for date in dates:
            datesZERO[date], unitsZERO[date] = zeroUnits(dTN[date], dTargs[date], dSC_[date], date)

    dates = []
    units = []
    
    count=0
    for d in datesZERO.keys():
        if len(unitsZERO[d]) > 0:
            for i in np.unique(unitsZERO[d]):
                dates.append(d)
                units.append(i)

    
    dfZERO = pd.DataFrame({'d':dates, 'unit': units})
    
    '''Saving values to .pkl'''     
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = dfZERO
    filename = subject+'_zeroUNITS.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()




#%%
'#############################################################################'
'            Decoder Population - Factor Analysis '
'#############################################################################'

dates = date_list

'Time of Trials'
tBL_list = [dTime[d]['BL'] for d in dates]
tPE_list = [dTime[d]['PE'] for d in dates]

'Distance'
TN = [dTN[d] for d in dates]
TL = [dTargs[d] for d in dates]
DX = [dDist[d] for d in dates]


'Spike Counts + Trial Information'
dSC     = [ dSC_[d] for d in dates]
targets = TL#[targ[d] for d in dates]


dTimes_ = {}
dDist_  = {}
dEV = {}

numUnits = []

for d, date in enumerate(dates):#[38,41,48]:#range(len(common_dates)):
   
    dEV[d]     = {}
    dTimes_[d] = {}
    dDist_[d]  = {}

    for k in ['BL', 'PE']:
        
        dEV[d][k] = {'shared':[], 'private':[], 'loadings':[]}
     
        if k == 'BL':
            times = tBL_list[d]
        elif k == 'PE':
            times = tPE_list[d]
        
        
        'Trials and their Target Locations'
        trials = TN[d][k]#.tolist()
        targs  = np.array(targets[d][k])
        dfUpdates = pd.DataFrame({'trials': trials, 'targs':targs})
        
        'Number of Units'
        numU = np.shape(dSC[d][k])[1]

        
        'Determine the indices of the spike count rows that below to each trial (and corresponding target location).'
        inds = []
        targ_inds = []
        for t in np.unique(dfUpdates['trials']):
            temp = dfUpdates.loc[dfUpdates['trials']==t].index.tolist()
            inds.append(temp)
            targ_inds.append(dfUpdates['targs'][temp[0]])
       
        'Determine the minimum number of trials for a given target location.'
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
            print(d, j, minNum)
            set_targ_inds = (TargInds[:,j]).astype(int).tolist()
            temp = np.concatenate([inds[i] for i in set_targ_inds]).tolist()
            sc = np.array(dSC[d][k])[temp,:]

            units_to_drop = dfZERO.loc[dfZERO['d']==date, 'unit'].tolist()
            sc_fixed = np.delete(sc, units_to_drop, axis=1)

            nS, nU = np.shape(sc_fixed)
            numUnits.append(nS)

            'Factor Analsysis'
            
            if np.any(np.isnan(np.sum(stats.zscore(sc_fixed, axis=0), axis=0))) == 1:
                print("WEIRD ERROR", d)
                temp_zero = np.where(np.sum(sc_fixed, axis=0) == 0)[0][0]
                print(temp_zero)
                sc_fixed = np.delete(sc_fixed, temp_zero, axis=1)
                print(np.shape(sc_fixed))
            
            results = varianceSharedPrivate_byTrial(stats.zscore(sc_fixed, axis=0)) 
            dEV[d][k]['shared'].append(results[0])
            dEV[d][k]['private'].append(results[1])
            dEV[d][k]['loadings'].append(results[2]) #axis=0: loadings, axis=1: factors; n_components, n_features


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
obj_to_pickle = [deg_list, dates, dEV, dTimes_, dDist_, numUnits]
filename = subject+'_FA_loadings_40sets_noTC.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()