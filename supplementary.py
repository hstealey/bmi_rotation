# -*- coding: utf-8 -*-



"""
______________________________________________________________________________

Figure 3 V1 - NEURAL LINE PLOT -  Population Shared Variance
______________________________________________________________________________


"""


"""
Created on Mon Mar 27 19:44:17 2023
Last Update: Friday, April 7, 2023

@author: hanna
"""

"""

Purpose of this Script



-Identify sessions for each animal that have washout.
-Plot speed (magnitude)

"""


import os
import glob
import tables
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
from   scipy import stats
import matplotlib.pyplot as plt
from   numpy.linalg import norm
from   collections import Counter
from   numpy import concatenate as ct 

import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.sans-serif': 'Arial'})

# ''' #! Change directory to BMI PYTHON location.'''
# os.chdir(r'C:\Users\hanna\OneDrive\Documents\bmi_python-bmi_developed')
# import riglib

''' #! Change directory to CUSTOM FUNCTIONS location.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')
from tuningCurve_fxns import performFA, spikeCount, tuningCurve, tuningCurveFA
from behavior_fxns    import plotTimes, basicCursorTimesPLOT
#from generatePickles_fxns import AmountofLearning2, cursorPathLength
from generatePickles_fxns import cursorPathLength



targRad   = {'A':[], 'B':[]}
rewardLen = {'A':[], 'B':[]}
numUnits  = {'A':{}, 'B':{}}


for S in ['\Airport', '\Brazos']:
    
    sub = S[1]
    os.chdir(r'C:\BMI_Rotation\Data'+S)
    degFolders = glob.glob('*')

    for degFolder in degFolders:
        os.chdir(r'C:\BMI_Rotation\Data'+S+'\\'+degFolder)
        dates = glob.glob('*')
        
        numBlock[sub][degFolder]  = []
        numTrials[sub][degFolder] = []
        numUnits[sub][degFolder]  = []
        
        for d in dates:
            os.chdir(r'C:\BMI_Rotation\Data'+S+'\\'+degFolder+'\\'+d)
            hdf = tables.open_file(glob.glob('*.hdf')[0])
            decoder_files = glob.glob('*.pkl')
            decoder_filename = decoder_files[0]
            KG_picklename = decoder_files[1]
            
            targRad[sub].append(hdf.root.task._v_attrs['target_radius'])
            rewardLen[sub].append(hdf.root.task._v_attrs['reward_time'])
            numUnits[sub][degFolder].append(len(u))


            ''' Determine units used in decoder. '''
            kf = pickle.load(open(decoder_filename, 'rb'), encoding='latin1')
            u = kf.units
            
            unitCode = {2: 'a',
                        4: 'b',
                        8: 'c',
                        16: 'd'}
            
            units = [str(u[i,0])+unitCode[u[i,1]] for i in range(len(u))]
            
            
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
            msg = hdf.root.task_msgs[:]['msg'] #b'wait', b'premove', b'target', b'hold', b'targ_transition', b'target', b'hold', b'targ_transition', b'reward'
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
            
            
            if len(np.unique(block_type)) == 2:
                keys = ['BL', 'PE']
                numB = 2
            if len(np.unique(block_type)) == 3:
                keys = ['BL', 'PE', 'PL']
                numB = 3
            elif len(np.unique(block_type)) == 4:
                keys = ['BL', 'PE', 'PL', 'WO']
                numB = 4
            else:
                print("ERROR IN NUMBER OF BLOCKS")
                
            numBlock[sub][degFolder].append(numB)
            
            numTrials[sub][degFolder].append(df.loc[df['errorClamp']==0, 'blockType'].values)
  
#%%
'___________________________'

'Number of Decoder Units'
'___________________________'

for S in ['A', 'B']:
    print(S)

    u50 = np.concatenate((numUnits[S]['neg50'], numUnits[S]['50']))        
    m50 = np.mean(u50)
    s50 = np.std(u50)
    se50 = stats.sem(u50)
    
    print(np.sum(u50))
    
    
    u90 = np.concatenate((numUnits[S]['neg90'], numUnits[S]['90']))        
    m90 = np.mean(u90)
    s90 = np.std(u90)
    se90 = stats.sem(u90)
    
    print(np.sum(u90))
    print('\t90', m90, s90)
    
    print(stats.ttest_ind(u50, u90))
    
#%%    
'___________________________'

'Number of Blocks Completed'
'___________________________'

for S in ['A', 'B']:
    print(S)
    for d in degFolders:
        print('\t', d, numBlock[S][d])
        print(d, Counter(numBlock[S][d]))
        
#%%
'__________________________________________________'

'Number of Trials Completed in Block 3 and Block 4'
'__________________________________________________'

for S in ['A', 'B']:
    print(S)

    tot3_50 = []
    tot4_50 = []
    
    for i in numTrials[S]['neg90']:
        tot3_50.append(len(np.where(i==3)[0]))
        tot4_50.append(len(np.where(i==4)[0]))
    
    for i in numTrials[S]['90']:
        tot3_50.append(len(np.where(i==3)[0]))
        tot4_50.append(len(np.where(i==4)[0]))
        
    print(np.mean(np.concatenate((tot3_50, tot4_50))))
    print(np.std(np.concatenate((tot3_50, tot4_50))))

#%%

'___________________________'

'Excluded decoder units'
'___________________________'


dSubjects  = {0:['airp','Subject A'], 1:['braz','Subject B']}
targetDeg = np.arange(0,360,45)

subject_ind = 1
subject = dSubjects[subject_ind][0]
os.chdir(r'C:\BMI_Rotation')

filename = subject+'_FA_units_failed.pkl'
open_file = open(filename, "rb")
loaded = pickle.load(open_file)
ZERO_dates, dfZERO = loaded
open_file.close()

print(len(ZERO_dates))

print(len(dfZERO))

z = np.array([Bev[0][d] for d in dfZERO['d'].values.tolist()])
print( len(np.where(z == '50')[0]) )
print( len(np.where(z == 'neg50')[0]) )

print( len(np.where(z == '90')[0]) )
print( len(np.where(z == 'neg90')[0]) )


#%%
"""
_________________________________________

Adaptation above BL (0)
_________________________________________

"""
S = 'Subject B'

print(np.where( np.array(dComp[S]['bMR']) >=0)[0])

for i in temp:
    print(Bev[0][i])
    

    
