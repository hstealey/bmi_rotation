# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 11:50:10 2023

@author: hanna
"""


import os
import glob
import pickle
import tables
import numpy as np
import pandas as pd
import seaborn as sb
import scipy as sp
from   scipy import stats

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")

'IBM Color Scheme'
blue    = [100/255, 143/255, 255/255]
purple  = [120/255, 94/255, 240/255]
magenta = [220/255, 38/255, 127/255]
orange  = [254/255, 97/255, 0/255]
yellow  = [255/255, 176/255, 0/255]


plt.rcParams.update({'font.sans-serif': 'Arial', 'lines.linewidth':1, 'lines.color':'k'})


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')

from behavioralMetric_fxns import getAdaptation, behaviorBL
from stats_fxns            import Ftest, sigSlope, cvLDA


'Adjustable Parameters'
c1 = magenta
c2 = yellow
c3 = blue

dMonkey = {'Subject A': 'Monkey A', 'Subject B': 'Monkey B'}
subject_list = ['Subject A', 'Subject B']
dDegs = {'50': 50, 'neg50': -50, '90': 90, 'neg90': -90}
dCols = {50:orange ,90:purple, -50:orange, -90:purple}



"""
_______________________________________________________________
_______________________________________________________________

    Decoder Unit Tuning Properties (over sets of 8 trials ("time"))
    ___________________________________________________    
    
    File Name:    subject+'_FA_loadings_40sets_noTC.pkl' 
    From Script:  Functions/generatePickles.py
    
    Parameters Saved:
             [0]    [1]   [2]   [3]     [4]     [5]            
      Sev =  degs, dates, dEV, dTimes, dDist, numUnits

________________________________________________________________
"""
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
fn = glob.glob('*_FA_loadings_40sets_noTC.pkl')
open_file = open(fn[0], "rb")
Aev = pickle.load(open_file)
open_file.close()

open_file = open(fn[1], "rb")
Bev = pickle.load(open_file)
open_file.close()

'___________________________________________________'



Adegs = [dDegs[d] for d in Aev[0]]
Bdegs = [dDegs[d] for d in Bev[0]]



"""
______________________________________________________________________________

Fraction of ADAPTATION & Single Unit Communality Over Time (sets)
______________________________________________________________________________
"""

'Behavior - BASELINE'
BL_behavior = {}
for Sev, subject in zip([Aev, Bev], ['Subject A', 'Subject B']):
    BL_behavior[subject] = behaviorBL(Sev)

'Behavior - ADAPTATION'
adt   = {}
dfADT = {}

for Sev, subject in zip([Aev, Bev], ['Subject A', 'Subject B']):
    adt[subject], dfADT[subject] = getAdaptation(Sev)


"""
______________________________________________________________________________

DATA DICTIONARIES - Behavioral Aaptation & Population Shared Variance 
______________________________________________________________________________


"""

dS = {}

for S, Sev, degs  in zip(subject_list, [Aev, Bev], [Adegs, Bdegs]):

    dS[S] = {'degs':degs, 'sBL':[], 'sPE':[], 'pBL':[], 'pPE':[]}

    for d in range(len(Sev[1])):
        
        'Assuming the total variance for each unit sums to 1.'
        sBL = [  np.sum(Sev[2][d]['BL']['loadings'][i]**2, axis=0) for i in range(40)]
        pBL = [1-np.sum(Sev[2][d]['BL']['loadings'][i]**2, axis=0) for i in range(40)]
        sPE = [  np.sum(Sev[2][d]['PE']['loadings'][i]**2, axis=0) for i in range(40)]
        pPE = [1-np.sum(Sev[2][d]['PE']['loadings'][i]**2, axis=0) for i in range(40)]
            
        dS[S]['sBL'].append(np.mean(sBL, axis=1))
        dS[S]['sPE'].append(np.mean(sPE, axis=1))
        dS[S]['pBL'].append(np.mean(pBL, axis=1))
        dS[S]['pPE'].append(np.mean(pPE, axis=1))


dComp = {'Subject A':{}, 'Subject B':{}}

for S, degs in zip(subject_list, [Adegs, Bdegs]):

    ind = np.arange(len(degs))
    
    dfIndDeg = pd.DataFrame({'ind':ind,'deg': np.abs(degs) })
    
    dComp[S]['b_degs']  = degs
    dComp[S]['b_ind']   = ind
    dComp[S]['b_ind50'] = dfIndDeg.loc[dfIndDeg['deg']==50, 'ind'].values.tolist()
    dComp[S]['b_ind90'] = dfIndDeg.loc[dfIndDeg['deg']==90, 'ind'].values.tolist()
    
    if S == 'Subject B':
        ind_ = ind
        degs_ = degs
        
        ind = np.delete(ind_, [38,41,48])
        degs = np.delete(degs_, [38,41,48])
        
    dfIndDeg = pd.DataFrame({'ind':ind,'deg': np.abs(degs) })
    
    dComp[S]['s_ind']   = ind
    dComp[S]['s_degs']  = degs
    dComp[S]['s_ind50'] = dfIndDeg.loc[dfIndDeg['deg']==50, 'ind'].values
    dComp[S]['s_ind90'] = dfIndDeg.loc[dfIndDeg['deg']==90, 'ind'].values

    'Behavior'
    maxIND = np.array(dfADT[S]['indMR'].values.tolist())
    ID     = np.array(dfADT[S]['ID'].values)
    IR     = np.array(dfADT[S]['IR'].values)
    MR     = np.array(dfADT[S]['MR'].values)
    n      = len(maxIND)
    adtS   = np.array([adt[S][i] for i in range(n)]).reshape((n,40))
     
    dComp[S]['maxIND']  = maxIND
    dComp[S]['bID']     = ID
    dComp[S]['bIR']     = IR
    dComp[S]['bMR']     = MR
    dComp[S]['rec']     = adtS
    
    'SV - Shared Variance'
    sBL  = np.array(dS[S]['sBL'])
    sPE  = np.array(dS[S]['sPE'])
    
    mBL = np.array([np.nanmean([sBL[i][j] if sBL[i][j] <0.9 else float('nan') for j in range(40)]) for i in range(len(sBL))])
    dComp[S]['mBL'] = mBL
    dComp[S]['sPE'] = sPE
    
    dComp[S]['sID'] = sPE[:,0]
    dComp[S]['sIR'] = sPE[:,1]
    dComp[S]['sMR'] = np.array([sPE[i][maxIND[i]] for i in range(len(maxIND))])    




#%%
"""
______________________________________________________________________________

SUPPLEMENTARY - BEHAVIOR - Trial Set Number of Maximum Adpatation
______________________________________________________________________________


"""

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((3,4)))
#fig.suptitle('Trial Set of Max Adaptation', fontsize=14)
fig.text(0.5, 0.0, 'Rotation Condition', ha='center', fontsize=12)
fig.text(0,0.5, 'TSN of MA Timepoint', fontsize=12, va='center', rotation=90)
plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)
for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    indMR_50 = dComp[S]['maxIND'][dComp[S]['b_ind50']]
    indMR_90 = dComp[S]['maxIND'][dComp[S]['b_ind90']]
    
    colors = [orange, purple]
    axis.set_title(dMonkey[S], fontsize=12)
    for i, r in enumerate([indMR_50, indMR_90]):
        axis.boxplot(r, widths=(0.35), positions=[i], 
                showfliers=True,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='k', alpha=1),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color='g', markeredgecolor='k'),
                medianprops=dict(color='k'),
                )
    
    axis.spines[['right', 'top']].set_visible(False)
    axis.set_xticks([0,1])
    axis.set_ylim([0,44])
    #axis.set_xticklabels(['50'+chr(176), '90'+chr(176)])
    axis.set_xticklabels(['Easy', 'Hard'])
    
    ratio = np.var(indMR_50, ddof=1)/np.var(indMR_90, ddof=1)
    
    # F1, p1 = Ftest(indMR_50, indMR_90)
    # F2, p2 = Ftest(indMR_90, indMR_50)
    
    if (ratio > 3) or (1/ratio > 3):
        equal_var = False
    else:
        equal_var = True
        
    t,p = stats.ttest_ind(indMR_50, indMR_90, equal_var=equal_var, alternative='two-sided')
    #print('\t F: {:.4f}/{:4f}'.format(F1, F2))
    #print('\t t = {:.4f} ({:.4e})'.format(t,p))
    
    #star = 'P = {:.4f}'.format(p)
    
    print(t,p, equal_var)
    star = 'n.s.'
    
    y1, y2 = [40.5, 41.0] 
    x1, x2 = [0, 1]
    
    #axis.plot([x2, x2],[y1, y2], color='k')
    axis.plot([x1, x2],[y2, y2], color='k')
    #axis.plot([x1, x1],[y1, y2], color='k')
    
    axis.text(0.35, 41.5, star, fontsize=10, fontstyle='italic')

#%%
"""
______________________________________________________________________________

SUPPLEMENTARY - BEHAVIOR - Rate of Adaptation (ID -> IA)
______________________________________________________________________________


"""



fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((3,4)))
fig.text(0.55, 0.0, 'Rotation Condition', ha='center', fontsize=12)
fig.text(0,0.5, 'Adapatation from ID to IA', fontsize=12, va='center', rotation=90)
plt.subplots_adjust(left=0.175,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)

for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    ind50 = dComp[S]['b_ind50']
    ind90 = dComp[S]['b_ind90']

    num = dComp[S]['bIR'] - dComp[S]['bID']
    
    indMR_50 = num[ind50]#/dComp[S]['maxIND'][ind50]
    indMR_90 = num[ind90]#/dComp[S]['maxIND'][ind90]
    
    colors = [orange, purple]
    axis.set_title(dMonkey[S], fontsize=12)
    for i, r in enumerate([indMR_50, indMR_90]):
        axis.boxplot(r, widths=(0.35), positions=[i], 
                showfliers=True,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='k', alpha=1),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color='g', markeredgecolor='k'),
                medianprops=dict(color='k'),
                )
    axis.spines[['right', 'top']].set_visible(False)
    
    ratio = np.var(indMR_50, ddof=1)/np.var(indMR_90, ddof=1)
    # F1, p1 = Ftest(indMR_50, indMR_90)
    # F2, p2 = Ftest(indMR_90, indMR_50)
    
    if (ratio > 3) or (1/ratio > 3):
        equal_var = False
    else:
        equal_var = True
    
    
        
    t,p = stats.ttest_ind(indMR_50, indMR_90, equal_var=equal_var, alternative='two-sided')
    # print('\t F: {:.4f}/{:4f}'.format(F1, F2))
    # print('\t t = {:.4f} ({:.4e})'.format(t,p))
    
    y1, y2 = [1.6, 1.7] #y1, y2 = [0.225, 0.23] 
    x1, x2 = [0, 1]
    
    print('{} - t: {:.4f} ({:.4e}) {}'.format(S, t, p, equal_var))
    print(len(indMR_50) + len(indMR_90) - 2)
    
    #axis.plot([x2, x2],[y1, y2], color='k')
    axis.plot([x1, x2],[y2, y2], color='k')
    #axis.plot([x1, x1],[y1, y2], color='k')
    
    #star = 'P = {:.4e}'.format(p)
    star = 'n.s.'
    #axis.text(-0.1, 0.235, star, fontsize=10, fontstyle='italic')
    axis.text(0.35, 1.75, star, fontsize=10, fontstyle='italic')


ax[0].set_ylim([-1,2])#ax[0].set_ylim([0,0.25])
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Easy', 'Hard'])



#%%
"""
______________________________________________________________________________

SUPPLEMENTARY - BEHAVIOR - Rate of Adaptation (ID -> MA)
______________________________________________________________________________


"""



fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((3,4)))
fig.text(0.55, 0.0, 'Rotation Condition', ha='center', fontsize=12)
fig.text(0,0.5, 'Adapatation per TSN', fontsize=12, va='center', rotation=90)
plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)
for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    ind50 = dComp[S]['b_ind50']
    ind90 = dComp[S]['b_ind90']

    num = dComp[S]['bMR'] - dComp[S]['bID']
    
    indMR_50 = num[ind50]/dComp[S]['maxIND'][ind50]
    indMR_90 = num[ind90]/dComp[S]['maxIND'][ind90]
    
    colors = [orange, purple]
    axis.set_title(dMonkey[S], fontsize=12)
    for i, r in enumerate([indMR_50, indMR_90]):
        axis.boxplot(r, widths=(0.35), positions=[i], 
                showfliers=True,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='k', alpha=1),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color='g', markeredgecolor='k'),
                medianprops=dict(color='k'),
                )
    axis.spines[['right', 'top']].set_visible(False)
    
    ratio = np.var(indMR_50, ddof=1)/np.var(indMR_90, ddof=1)
    # F1, p1 = Ftest(indMR_50, indMR_90)
    # F2, p2 = Ftest(indMR_90, indMR_50)
    
    if (ratio > 3) or (1/ratio > 3):
        equal_var = False
    else:
        equal_var = True
        
    # print(equal_var)
        
    t,p = stats.ttest_ind(indMR_50, indMR_90, equal_var=equal_var, alternative='two-sided')
    #print(t,p, equal_var)
    #print('\t F: {:.4f}/{:4f}'.format(F1, F2))
    #print('\t t = {:.4f} ({:.4e})'.format(t,p))
    
    print('{} - t: {:.4f} ({:.4e}) {}'.format(S, t, p, equal_var))
    print(len(indMR_50) + len(indMR_90) - 2)
    
    
    y1, y2 = y1, y2 = [0.225, 0.23] 
    x1, x2 = [0, 1]
    
    #axis.plot([x2, x2],[y1, y2], color='k')
    axis.plot([x1, x2],[y2, y2], color='k')
    #axis.plot([x1, x1],[y1, y2], color='k')
    
    #star = 'P = {:.4e}'.format(p)
    star = 'n.s.'
    axis.text(0.35, 0.235, star, fontsize=10, fontstyle='italic')


ax[0].set_ylim([0,0.25])
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Easy', 'Hard'])


#%%
"""
______________________________________________________________________________

SUPPLEMENTARY - BEHAVIOR - Rate of Adaptation - NEURAL VARIATION (ID -> IA)
______________________________________________________________________________


"""



fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((3,4)))
fig.text(0.55, 0.0, 'Rotation Condition', ha='center', fontsize=12)
fig.text(0,0.5, 'Δ %sv from ID to IA', fontsize=12, va='center', rotation=90)
plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)
for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    ind50 = dComp[S]['s_ind50']
    ind90 = dComp[S]['s_ind90']

    num = dComp[S]['sIR'] - dComp[S]['sID']
    
    indMR_50 = num[ind50]#/dComp[S]['maxIND'][ind50]
    indMR_90 = num[ind90]#/dComp[S]['maxIND'][ind90]
    
    colors = [orange, purple]
    axis.set_title(dMonkey[S], fontsize=12)
    for i, r in enumerate([indMR_50, indMR_90]):
        axis.boxplot(r, widths=(0.35), positions=[i], 
                showfliers=True,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='k', alpha=1),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color='g', markeredgecolor='k'),
                medianprops=dict(color='k'),
                )
    axis.spines[['right', 'top']].set_visible(False)
    
    ratio = np.var(indMR_50, ddof=1)/np.var(indMR_90, ddof=1)
    # F1, p1 = Ftest(indMR_50, indMR_90)
    # F2, p2 = Ftest(indMR_90, indMR_50)
    
    if (ratio > 3) or (1/ratio > 3):
        equal_var = False
    else:
        equal_var = True
        

    t,p = stats.ttest_ind(indMR_50, indMR_90, equal_var=equal_var, alternative='two-sided')
    #print('\t F: {:.4f}/{:4f}'.format(F1, F2))
    print('\t t = {:.4f} ({:.4e})'.format(t,p))
    print(len(indMR_50)+len(indMR_90)-2, equal_var)
    #print(t,p, equal_var)
    y1, y2 = [0.065, 0.075] 
    x1, x2 = [0, 1]
    
    #axis.plot([x2, x2],[y1, y2], color='k')
    axis.plot([x1, x2],[y2, y2], color='k')
    #axis.plot([x1, x1],[y1, y2], color='k')
    
    #star = 'P = {:.4e}'.format(p)
    star = 'n.s.'
    axis.text(0.35, 0.078, star, fontsize=10, fontstyle='italic')


ax[0].set_ylim([-0.05,0.085])
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Easy', 'Hard'])

#%%
"""
______________________________________________________________________________

SUPPLEMENTARY - BEHAVIOR - Rate of Adaptation - NEURAL VARIATION (ID -> MA)
______________________________________________________________________________


"""



fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((3,4)))
fig.text(0.55, 0.0, 'Rotation Condition', ha='center', fontsize=12)
fig.text(0,0.5, 'Δ %sv per TSN', fontsize=12, va='center', rotation=90)
plt.subplots_adjust(left=0.25,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)
for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    ind50 = dComp[S]['s_ind50']
    ind90 = dComp[S]['s_ind90']

    num = dComp[S]['sMR'] - dComp[S]['sID']
    
    indMR_50 = num[ind50]/dComp[S]['maxIND'][ind50]
    indMR_90 = num[ind90]/dComp[S]['maxIND'][ind90]
    
    colors = [orange, purple]
    axis.set_title(dMonkey[S], fontsize=12)
    for i, r in enumerate([indMR_50, indMR_90]):
        axis.boxplot(r, widths=(0.35), positions=[i], 
                showfliers=True,
                notch=True, patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='k', alpha=1),
                capprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                flierprops=dict(color='g', markeredgecolor='k'),
                medianprops=dict(color='k'),
                )
    axis.spines[['right', 'top']].set_visible(False)
    
    ratio = np.var(indMR_50, ddof=1)/np.var(indMR_90, ddof=1)
    # F1, p1 = Ftest(indMR_50, indMR_90)
    # F2, p2 = Ftest(indMR_90, indMR_50)
    
    if (ratio > 3) or (1/ratio > 3):
        equal_var = False
    else:
        equal_var = True
        
    
    t,p = stats.ttest_ind(indMR_50, indMR_90, equal_var=equal_var, alternative='two-sided')
   # print(t,p,equal_var)
    # print('\t F: {:.4f}/{:4f}'.format(F1, F2))
    print('\t t = {:.4f} ({:.4e})'.format(t,p))
    print(equal_var)
    
    y1, y2 = [0.008, 0.0085] 
    x1, x2 = [0, 1]
    
    #axis.plot([x2, x2],[y1, y2], color='k')
    axis.plot([x1, x2],[y2, y2], color='k')
    #axis.plot([x1, x1],[y1, y2], color='k')
    
    #star = 'P = {:.4e}'.format(p)
    star='n.s.'
    axis.text(0.375, 0.00875, star, fontsize=10, fontstyle='italic')


ax[0].set_ylim([-0.005,0.01])
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Easy', 'Hard'])


#%%

"""
________________________________________________________________________
Supplementary Figure 5
Cross Validation (How the number of folds and iterations impact results) 
________________________________________________________________________
"""

test_folds = [2,3,4,5,10]
test_iters = [1,10,100,200,500,1000]


dKN = {'Subject A': {},
        'Subject B': {}}

dKN_results = {'Subject A': {},
               'Subject B': {}}


'Compute model and chance accuracy.'
for S in subject_list:
    
    ind = dComp[S]['s_ind']
    deg = np.abs(dComp[S]['s_degs'])
    
    for k in ['mBL', 'sID', 'sIR', 'sMR']:
        dKN[S][k] = {}
        dKN_results[S][k] = {}
        
        for tf in test_folds:
            dKN[S][k][tf] = {}
            dKN_results[S][k][tf] = {}
            
            for ti in test_iters:
                
                X = np.array(dComp[S][k][ind]).reshape(-1,1)
                y = deg
  
                scoresMODEL, _ = cvLDA(dComp[S], X, y, tf, ti)
        
                dKN_results[S][k][tf][ti] = scoresMODEL
                dKN[S][k][tf][ti] = stats.sem(scoresMODEL)
     
        
                print(S, k, tf, ti, dKN[S][k][tf][ti])
        

#%%
# S = 'Subject A'

dColors = {2:'red',3:yellow,4:'k',5: blue,10: purple}


#plt.legend()

#width = 0.75
#x_loc = [0, width, width*2, width*3, width*4]

fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, sharey=True, figsize=(6,12))
fig.text(0,0.5, 'MODEL SEM', fontsize=18, fontweight='bold', rotation=90, va='center')
fig.text(0.5, 0, 'Number of Iterations', fontsize=18, fontweight='bold', ha='center')
plt.subplots_adjust(left=0.125,
                bottom=0.05,
                right=0.9,
                top=1,
                wspace=0.15,
                hspace=0.15)

dLabelK = {'mBL': 'mBL', 'sID': 'ID', 'sIR': 'IA', 'sMR': 'MA'}
axA = [ax[0][0], ax[1][0], ax[2][0], ax[3][0]]
axB = [ax[0][1], ax[1][1], ax[2][1], ax[3][1]]

for S, axis_list in zip(subject_list, [axA, axB]):
    for k, axis in zip(['mBL', 'sID', 'sIR', 'sMR'], axis_list):
        for tf in test_folds:
            axis.plot(test_iters, [dKN[S][k][tf][ti] for ti in test_iters], marker='o', label=tf, color=dColors[tf])
            axis.spines[['right', 'top']].set_visible(False)
            if S == 'Subject A':
                axis.set_ylabel(dLabelK[k])

ax[0][1].legend(title='k')
ax[3][0].set_xticks(test_iters)
ax[3][0].set_xticklabels(test_iters, rotation=90)           
ax[3][1].set_xticklabels(test_iters, rotation=90)  


#%%

# S = 'Subject A'
# k = 'mBL'#, 'sID', 'sIR', 'sMR']
# ti = test_iters[-1]

# scores = []
# labels = []
# for tf in test_folds:
#     scores.append(dKN_results[S][k][tf][ti])
#     labels.append([str(tf)]*ti*tf)

# #%%

# 'One-Way ANOVA: Number of Folds for Maximum Iterations'

# df = pd.DataFrame({'scores': np.concatenate((scores)), 'labels': np.concatenate((labels))})

# model = ols('scores ~ C(labels)', data=df).fit()

# print('\t', sm.stats.anova_lm(model, typ=1))


# tukey = pairwise_tukeyhsd(endog=df['scores'],
#                           groups=df['labels'],
#                           alpha=0.05)


# print(tukey)






