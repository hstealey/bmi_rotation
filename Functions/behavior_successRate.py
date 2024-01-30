# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:49:25 2023

@author: hanna
"""

import os
import glob
import tables
import pickle



import tqdm

import pandas as pd
import numpy as np
from scipy import stats

import seaborn as sb
import matplotlib.pyplot as plt

from collections import Counter


dSubjects  = {0:['airp','Airport'], 1:['braz','Brazos']}
targetDeg = np.arange(0,360,45)

subject_ind = 1
subject = dSubjects[subject_ind][0]
subj = dSubjects[subject_ind][1]
root_path = r'C:\BMI_Rotation\Data'+'\\'+subj

subject_list = ['Subject A', 'Subject B']

'IBM Color Scheme'
blue    = [100/255, 143/255, 255/255]
purple  = [120/255, 94/255, 240/255]
magenta = [220/255, 38/255, 127/255]
orange  = [254/255, 97/255, 0/255]
yellow  = [255/255, 176/255, 0/255]

#%%
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
filename = glob.glob('*_behavior_successRate.pkl')
open_file = open(filename[0], "rb")
loaded = pickle.load(open_file)
dAttemptsA, rotA = loaded
open_file.close()

open_file = open(filename[1], "rb")
loaded = pickle.load(open_file)
dAttemptsB, rotB = loaded
open_file.close()

dAttempts = {'Subject A': dAttemptsA, 'Subject B': dAttemptsB}

#%%
def getBehavior_success(path):
    
    os.chdir(path)
    
    hdf = tables.open_file(glob.glob('*.hdf')[0])
    
    max_allowed_attempts = hdf.root.task._v_attrs['max_attempts']
    rotation_angle = np.abs(hdf.root.task._v_attrs['rot_angle_deg'])
    
    
    '''Parse HDF file.'''    
    msg = hdf.root.task_msgs[:]['msg'] 
    ind = hdf.root.task_msgs[:]['time']
    
    #pert = hdf.root.task[:]['pert'][:,0]
    error_clamp = hdf.root.task[:]['error_clamp'][:,0]
    block_type  = hdf.root.task[:]['block_type'][:,0]
    
    
    target_x = hdf.root.task[:]['target'][:,0]
    target_y = hdf.root.task[:]['target'][:,2]
    
    
    angle = []
    for x,y in zip(target_x, target_y):
        
        if x == 0:
            angle_temp = 0
        else:
            angle_temp = np.round(np.arctan2(y,x)*(180/np.pi),2)
            
            if angle_temp < 0:
                angle_temp+=360
           
            
        angle.append(angle_temp)
    
    
    
        
    bT = block_type[ind[:-1]]
    eC = error_clamp[ind[:-1]]
    deg = np.array(angle)[ind[:-1]]
    
    df = pd.DataFrame({'msg': msg[:-1],
                        'block_type': bT,
                        'error_clamp': eC,
                        'deg': deg})
    
    trial_num = []
    
    
    
    count=1
    for m in df.msg:
        if m == b'reward':
            trial_num.append(count)
            count+=1
        else:
            trial_num.append(count)
        
            
    df['trial_num'] = trial_num
    
    
    return(df, max_allowed_attempts, rotation_angle)
 

os.chdir(root_path)
degFolders = glob.glob('*')



dAttempts = {}
max_allowed_attempts = []
rotation_angle = []

d = 0

for degFolder in tqdm.tqdm(degFolders):
    os.chdir(root_path+'\\'+degFolder)
    dates = glob.glob('*')
    

    
    for date in dates:
        
        dAttempts[d] = {}
        
        
        path = os.path.join(root_path, degFolder, date)
        
        df, MAA, RA = getBehavior_success(path)
        max_allowed_attempts.append(MAA)
        rotation_angle.append(RA)

    
        print('Finished:', degFolder, date)


        for block_num in [1,2]:
            
            if block_num == 1:
                k = 'BL'
            elif block_num == 2:
                k = 'PE'
            
            

            dfBlock = df.loc[(df['block_type'] == block_num) & (df['error_clamp'] == 0)]
    
            deg_inds = []
            for deg in np.unique(dfBlock['deg']):
                deg_inds_ = dfBlock.loc[ (dfBlock['msg'] == b'reward') &(dfBlock['deg'] == deg)].index.tolist()[:40]
                deg_inds.append(deg_inds_)
    
            maxInd = np.max(deg_inds)  


            fail1 = 0
            fail1_trialNums = []
            fail2 = 0
            fail2_trialNums = []
            
            for tp in dfBlock.loc[dfBlock['msg'] == b'timeout_penalty'].index:
                if tp > maxInd:
                    pass
                else:
                    fail1+=1
                    fail1_trialNums.append(dfBlock.loc[tp]['trial_num'])
                    
            for tp in dfBlock.loc[dfBlock['msg'] == b'hold_penalty'].index:
                if tp > maxInd:
                    pass
                else:
                    fail2+=1
                    fail2_trialNums.append(dfBlock.loc[tp]['trial_num'])
            
            
            fail = np.concatenate((fail1_trialNums, fail2_trialNums))
            
            #dFail1 = Counter(fail1_trialNums)
            #dFail2 = Counter(fail2_trialNums)
            
            dFail = Counter(fail)
            dFail_keys = dFail.keys()
            
            if block_num == 1:
                attempts_TN = np.arange(1,320+1)
            elif block_num == 2:
                attempts_TN = np.arange(385,385+320+1)
            
            attempts = np.ones(len(attempts_TN))
            
            for i, ind in enumerate(attempts_TN):
                if ind in dFail.keys():
                    attempts[i]+=dFail[ind]
            
            dAttempts[d][k] = attempts
            
            if np.max(attempts) > 10:
                print(d)
                print(dFail)
            
        d+=1


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
obj_to_pickle = [dAttempts, rotation_angle]
filename = subject+'_behavior_successRate.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()
        
            
                    
#%%
"""
"""

print('Difference in Max Attempts')
for S in subject_list:
    dA = dAttempts[S]
    
    maxAttemptsBL = []
    maxAttemptsPE = []
    
    for d in range(len(dA)):
        BL = dA[d]['BL']
        PE = dA[d]['PE']
        
        maxAttemptsBL.append(np.max(BL))
        maxAttemptsPE.append(np.max(PE))
    
    t,p = stats.ttest_ind(BL, PE)
    
    
    
    print('\t{}: t={:.3f} ({:.2e})'.format(S, t, p))
        
        
#%%

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((7,3)))
#fig.suptitle('Average Behavioral Adaptation over a Single Session', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Trial Set #', ha='center', fontsize=14, fontweight='bold')
#fig.text(0, 0.50 , 'Adaptation (%)'.format(chr(176)), va='center', fontsize=14, fontweight='bold', rotation=90)
plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)


for S, rotation_angle, axis in zip(subject_list, [rotA, rotB], [ax[0], ax[1]]):
    
    rot = np.array([int(i) for i in rotation_angle])
    ind50 = np.where(rot == 50)[0]
    ind90 = np.where(rot == 90)[0]
    
    dA = dAttempts[S]
    
    BL = []
    PE = []
    
    BL50 = []
    BL90 = []
    
    PE50 = []
    PE90 = []
    
    for d in range(len(dA)):
        BL.append(dA[d]['BL'])
        PE.append(dA[d]['PE'])
        
        if d in ind50:
            BL50.append(dA[d]['BL'])
            PE50.append(dA[d]['PE'])
        
        if d in ind90:
            BL90.append(dA[d]['BL'])
            PE90.append(dA[d]['PE'])
        
    BL50 = np.concatenate((BL50))
    print('{}, BL50 - {:.4f}%'.format(S, 100*np.shape(np.where(BL50 <= 2))[1]/ np.shape(BL50)[0]))
    
    
    PE50 = np.concatenate((PE50))
    print('{}, PE50 - {:.4f}%'.format(S, 100*np.shape(np.where(PE50 <= 2))[1]/ np.shape(PE50)[0]))
    
    
    BL90 = np.concatenate((BL90))
    print('{}, BL90 - {:.4f}%'.format(S, 100*np.shape(np.where(BL90 <= 2))[1]/ np.shape(BL90)[0]))
    

    PE90 = np.concatenate((PE90))
    print('{}, PE90 - {:.4f}%'.format(S, 100*np.shape(np.where(PE90 <= 2))[1]/ np.shape(PE90)[0]))
    
    
    
    lenBL50 = len(BL50)
    lenBL90 = len(BL90)
    lenPE50 = len(PE50)
    lenPE90 = len(PE90)
    
    
    lenBL = lenBL50 + lenBL90
    lenPE = lenPE50 + lenPE90
    
    deg = ['50']*lenBL50 + ['90']*lenBL90 + ['50']*lenPE50 + ['90']*lenPE90
    block =['BL']*lenBL + ['PE']*lenPE
    

    df = pd.DataFrame({'data': np.concatenate((BL50, BL90, PE50, PE90)),
                        'block': block, 'deg': deg})
    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    model = ols('data ~ C(block) + C(deg) + C(block):C(deg)', data=df).fit()
    
    # print(S)
    # print('\tMean BL50: {:.5f} ({:.5f})'.format(np.mean(BL_50), np.std(BL_50)))
    # print('\tMean PE50: {:.5f} ({:.5f})'.format(np.mean(PE_50), np.std(PE_50)))
    # print('\tMean BL90: {:.5f} ({:.5f})'.format(np.mean(BL_90), np.std(BL_90)))
    # print('\tMean PE90: {:.5f} ({:.5f})'.format(np.mean(PE_90), np.std(PE_90)))
    # print(sm.stats.anova_lm(model, typ=2))
    # print(len(df))

    # # import scipy as sp
    # # res = sp.stats.tukey_hsd(BL_50, BL_90, PE_50, PE_90)
    # # print(res)
    
    
    # print(stats.ttest_ind(PE50, PE90))
    
    # print(np.mean(PE50) - np.mean(PE90))

    from statsmodels.stats.multicomp import MultiComparison
    
    mc = MultiComparison(data=df['data'], groups=df['deg'])
    tukey_results = mc.tukeyhsd()
    
    print(tukey_results.summary())










