# -*- coding: utf-8 -*-

"""
______________________________________________________________________________________________
Purpose: This script is designed to visually explore the relationship of 
         variability in neural decoder population spiking activity 
         (as determined by FACTOR ANALYSIS) and behavior (adaptation to rotation perturbation).


PRE-REQS:
        Decoder Unit Factor Loadings; Behavior (Distance, Time)
        File Name:    subject+'_FA_loadings_40sets.pkl'
        From Script:  Neural-And-Behavior-Relationships.py           
            Sev =  degs, dates, dTC, dEV, dTimes, dDist, subject

______________________________________________________________________________________________
        
"""


"""
    
______________________________________________________________________________________________


ADDITIONAL CUSTOM FUNCTIONS:

    [1] Functions/behavioralMetric_fxns.py: 
        
        Purpose: to return behavioral metric ("adaptation": combination of distance, time), 
                 baseline behavior (trial time, cursor path lengths) &
                 decoder unit communality from FA
        
        __________________
        getAdaptation
            input: Sev
            outputs: adt, dfADT
     
        __________________
        behaviorBL
            input:
            output: 
    
        
    [2] Functions/stats_fxns
    
        Purpose: to validate built-in stats functions

        __________________
        Ftest (determines if the variances of variable 1 (v1) and variable 2 (v2) are significantly different)
            inputs: v1, v2
            outputs: (no return)
        
        __________________
        sigSlope (determines if a linear regression slope is significantly different from 0)
            inputs: X, y, alpha, returnSE=FALSE
            outputs: slope, intercept, p, LL, UL
        
        __________________
        cvLDA
    

created on: Thu Dec  1 19:58:06 2022
last update: July 15, 2023

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
from stats_fxns            import Ftest, sigSlope#, cvLDA


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




#%%


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


#%%
'Mean baseline behavioral metrics (time, distance) were consistent between 50 and 90.'

for S in subject_list:
    ind50 = dComp[S]['s_ind50']
    ind90 = dComp[S]['s_ind90']
    
    t50 = [np.mean(BL_behavior[S][d]['time']) for d in ind50]
    t90 = [np.mean(BL_behavior[S][d]['time']) for d in ind90]

    d50 = [np.mean(BL_behavior[S][d]['dist']) for d in ind50]
    d90 = [np.mean(BL_behavior[S][d]['dist']) for d in ind90]
    
    ratioT1 = np.var(t50, ddof=1)/np.var(t90, ddof=1)
    ratioT2 = np.var(t90, ddof=1)/np.var(t50, ddof=1)
    ratioD1 = np.var(d50, ddof=1)/np.var(d90, ddof=1)
    ratioD2 = np.var(d90, ddof=1)/np.var(d50, ddof=1)
 
    # print(S, ratioT1, ratioT2)
    # print(S, ratioD1, ratioD2)
    
    if S == 'Subject A':
        equal_var_time = True
        equal_var_dist = False
    elif S == 'Subject B':
        equal_var_time = False
        equal_var_dist = False
    else:
        print('ERROR')
    

    print(S)
    t, p = stats.ttest_ind(t50, t90, equal_var=equal_var_time)
    print('\ttime: {:.4f} ({:.4e}) - equal var?: {}'.format(t,p, equal_var_time))
    

    t, p = stats.ttest_ind(d50, d90, equal_var=equal_var_dist)
    print('\tdist: {:.4f} ({:.4e}) - equal var?: {}'.format(t,p, equal_var_dist))

#%%

"""
______________________________________________________________________________

Figure 2B - Average Behavior Over Sessions
______________________________________________________________________________


"""


m_list = []
rec_list = []

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((7,3)))
#fig.suptitle('Average Behavioral Adaptation over a Single Session', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Trial Set #', ha='center', fontsize=14, fontweight='bold')
fig.text(0, 0.50 , 'Adaptation (%)'.format(chr(176)), va='center', fontsize=14, fontweight='bold', rotation=90)
plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0)


dCols = {50: orange, 90: purple}

for S, Sev, axis, Sdegs in zip(['Subject A', 'Subject B'], [Aev, Bev], [ax[0], ax[1]], [Adegs, Bdegs]):

    Sdegs = np.abs(np.array(Sdegs))
    rec_array = np.array([-1*(adt[S][k]/adt[S][k][0]) for k in adt[S].keys()])
    
    for deg, lab in zip([50, 90],['Easy','Hard']):
        indDeg = list(np.where(Sdegs == deg)[0])
        rec_list.append(rec_array[indDeg])
        m = np.mean(rec_array[indDeg], axis=0)
        s = stats.sem(rec_array[indDeg], axis=0)
        axis.plot(m, c=dCols[deg], ls='-', lw=3, label='{} (n={})'.format(lab, len(indDeg)))
        axis.fill_between(np.arange(40), m-s, m+s, color=dCols[deg], alpha=0.25)
        axis.set_title(dMonkey[S], fontsize=14)
        axis.legend(title='Rotation Condition', loc='lower right', frameon=False, labelspacing=0.2, handlelength=1)
        m_list.append(m)
        
        
        
        def func(x, a, b, c):
            return(a * np.exp(-b*x) + c)
        
        xdata = np.arange(40)
        ydata = m
        params, _ = curve_fit(func, xdata, ydata)
        
        a,b,c = params
        
        y_func = func(xdata, a,b,c)
        #axis.plot( np.array(sigCorr[S])*10)
        axis.plot(np.arange(40), y_func, zorder=1, color='k', lw=2, ls='-')
        #axis.text(15, 8, '{:.2f} * exp(-{:.2f}*x) + {:.2f}'.format(a,b,c))
        y_hat = func(xdata,a,b,c)
        y_mean = np.mean(ydata)
    
    
        num = np.sum(np.subtract(y_hat, y_mean)**2)
        den = np.sum(np.subtract(ydata, y_mean)**2)
    
        r = np.sqrt(num/den)
        print(r)
        #axis.text(25, 10, 'r\u00b2: {:.2f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))
        print('r\u00b2: {:.2f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))
        
        
        axis.spines[['right', 'top']].set_visible(False)
    axis.set_xlabel('Trial Set Number', fontsize=14, fontweight='bold')
axis.set_xlim([-2,42])
axis.set_xticks(np.arange(0,41,5))
axis.set_xticklabels(np.arange(0,41,5))
axis.set_yticks(np.arange(-1,0.1,0.2))
axis.set_yticklabels(['ID', '20', '40', '60', '80', 'BL'])



#%%

"""
______________________________________________________________________________

Figure 2C - BEHAVIOR - SCATTER PLOTS
______________________________________________________________________________

"""

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(9,5))
#fig.suptitle('Behavioral Adaptation', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Behavioral Timepoint', ha='center', fontweight='bold',fontsize=14)
fig.text(0.0, 0.5, 'Adaptation', va='center', rotation=90,fontweight='bold', fontsize=14)
plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.0,
                    hspace=0.1)



left, bottom, width, height = [0.325, 0.35, 0.15, 0.2] 
ax2 = fig.add_axes([left, bottom, width, height])
ax3 = fig.add_axes([left+0.415, bottom, width, height])

ax[0].set_title(dMonkey['Subject A'], fontsize=14)
ax[1].set_title(dMonkey['Subject B'], fontsize=14)

ax[0].axhline(0, color='k', ls='-.')
ax[1].axhline(0, color='k', ls='-.')

ax[0].set_xlim([-0.5,2.5])
ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(['Initial\nDrop', 'Initial\nAdaptation', 'Max\nAdaptation'], fontsize=10)

ax[0].set_ylim([-3.0,0.5])
ax[0].set_yticks(np.arange(-3.0,0.1,0.5))
#ax[0].set_yticklabels(np.arange(-3.0,0.1,0.5), fontsize=12)
ax[0].set_yticklabels([ '-3','-2.5', '-2', '-1.5', '-1', '-0.5','BL'], fontsize=12)

for S, axis, axis_sub in zip(subject_list, [ax[0], ax[1]], [ax2, ax3]):
    
    m50 = np.zeros((3))
    m90 = np.zeros((3))
    
    s50 = np.zeros((3))
    s90 = np.zeros((3))
    
    axis.set_xlabel('Behavioral Timepoint', fontweight='bold', fontsize=14)
    
    
    for i,k in enumerate(['bID', 'bIR', 'bMR']):
        
        ind50 = dComp[S]['b_ind50']
        ind90 = dComp[S]['b_ind90']
    
        r1 = dComp[S][k][ind50]
        r2 = dComp[S][k][ind90]
        
        m50[i] = np.mean(r1)
        m90[i] = np.mean(r2)
    
        s50[i] = stats.sem(r1)
        s90[i] = stats.sem(r2)

        axis.scatter(np.repeat(i-0.09,len(r1)), r1, marker='o', color=orange, s=75, alpha=0.2)
        axis.scatter(np.repeat(i+0.09,len(r2)), r2, marker='^', color=purple, s=75, alpha=0.2)
        
        ratio = np.var(r1, ddof=1)/np.var(r2, ddof=1)
    
        if (ratio > 3) or (1/ratio > 3):
            equal_var = False
        else:
            equal_var = True
            
        t,p = stats.ttest_ind(r1, r2, equal_var=equal_var, alternative='two-sided')
        print('t:{:.4f} ({:.3f} - {})'.format(t,p,equal_var))
    
    axis.scatter([0,1,2], m50, color=orange, marker='o', s=150, edgecolor='k', label='Easy (n={})'.format(len(r1)))
    axis.scatter([0,1,2], m90, color=purple, marker='^', s=150, edgecolor='k', label='Hard (n={})'.format(len(r2)))
    axis.spines[['right', 'top']].set_visible(False)
    'Inset: scatter'
    plusMinus = {0:-1, 1:1}
    jitter50 = np.multiply([plusMinus[np.random.randint(0,2)] for i in range(len(r1))], np.random.random(len(r1))/55)
    jitter90 = np.multiply([plusMinus[np.random.randint(0,2)] for i in range(len(r2))], np.random.random(len(r2))/55)
    
    axis_sub.scatter(1.95 + jitter50, r1, color=orange, alpha=0.25, marker='.')
    axis_sub.scatter(2.05 + jitter90, r2, color=purple, alpha=0.25, marker='^', s=12)
    axis_sub.scatter(i, m50[-1], color=orange, edgecolor='k', marker='o', s=100)
    axis_sub.scatter(i, m90[-1], color=purple, edgecolor='k', marker='^', s=100)
    axis_sub.set_xlim([1.90,2.10])
    axis_sub.set_xticks([])
    axis_sub.set_ylim([-0.45,0.2])
    axis_sub.set_yticks([])
    axis_sub.axhline(0, color='k', ls='-.', zorder=0)

    axis.legend(title='Rotation Condition', loc='lower right', ncol=1)
    
ax[0].text(-0.1,0.1,'***', fontsize=16)
ax[0].text(0.9,0.1,'***', fontsize=16)
ax[0].text(1.9,0.1,'***', fontsize=16)

ax[1].text(0,0.2,'t.', fontsize=16, fontstyle='italic')
ax[1].text(1,0.2,'t.', fontsize=16, fontstyle='italic')
ax[1].text(1.9,0.1,'***', fontsize=16)


y = 0.13
for x1, x2 in zip([-0.25,0.75,1.75], [0.25, 1.25, 2.25]):
    ax[0].plot([x1, x2],[y, y], color='k')
    ax[1].plot([x1, x2],[y, y], color='k')

#%%

"""
______________________________________________________________________________

Figure 3A - NEURAL PLOTS - Average %sv adaptation profile over sessions
______________________________________________________________________________


"""

fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False, figsize=((4,8)))
#fig.suptitle('Average Changes in %sv Across Single Sessions', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Trial Set Number', ha='center', fontsize=12, fontweight='bold')
fig.text(0, 0.5, '%sv'.format(chr(176)), va='center', fontsize=18, fontweight='bold', rotation=90)

plt.subplots_adjust(left=0.15,
                    bottom=0.1,
                    right=0.95,
                    top=0.85,
                    wspace=0.15,
                    hspace=0.2)


for S, axis in zip(subject_list, [ax[0], ax[1]]):
    

        ind50 = dComp[S]['s_ind50']
        ind90 = dComp[S]['s_ind90']
        
        mBL = dComp[S]['mBL']
        sPE = dComp[S]['sPE']
        
        dSV = sPE


        m50BL, s50BL = np.mean(mBL[ind50]), stats.sem(mBL[ind50])
        m90BL, s90BL = np.mean(mBL[ind90]), stats.sem(mBL[ind90])
        
        m50, s50 = [np.mean(dSV[ind50,:],axis=0), stats.sem(dSV[ind50,:], axis=0)]
        m90, s90 = [np.mean(dSV[ind90,:],axis=0), stats.sem(dSV[ind90,:], axis=0)]


        axis.plot(m50, color=orange, lw=3, label='Easy (n={})'.format(len(ind50)))
        axis.fill_between(np.arange(40), m50-s50, m50+s50, color=orange, alpha=0.25)
        
        axis.plot(m90, color=purple, lw=3, label='Hard (n={})'.format(len(ind90)))
        axis.fill_between(np.arange(40), m90-s90, m90+s90, color=purple, alpha=0.25)
        
        axis.errorbar(-5, m50BL, yerr=s50BL, capsize=3, color=orange,markeredgecolor='k', marker='*', markersize=15)
        axis.errorbar(-5, m90BL, yerr=s90BL, capsize=3, color=purple, markeredgecolor='k', marker='*', markersize=15)
        #axis.errorbar(-5, m50BL, yerr=s50BL,  color='k', markeredgecolor='k', marker='*', markersize=10, label='Baseline', zorder=0)
        
        axis.set_title(dMonkey[S], fontsize=16)    
        axis.axvline(0, color='grey', ls='-.', label='Perturbation Onset')

        axis.legend(loc='lower right', title='Rotation Condition', ncol=1, frameon=False, labelspacing=0.2, handlelength=1)
        #leg = axis.legend(title='Rotation Condition',  ncol=2, loc='lower right', handlelength=0.75, labelspacing=0.1)
        #leg.legendHandles[3].set_color(orange)
        
        axis.spines[['right', 'top']].set_visible(False)
        
        for m_deg in [m50, m90]:
            def func(x, a, b, c):
                return(a * np.exp(-b*x) + c)
            
            xdata = np.arange(40)
            ydata = m_deg
            params, _ = curve_fit(func, xdata, ydata)
            
            a,b,c = params
            
            y_func = func(xdata, a,b,c)
            #axis.plot( np.array(sigCorr[S])*10)
            axis.plot(np.arange(40), y_func, zorder=1, color='k', lw=2, ls='-')
            #axis.text(15, 8, '{:.2f} * exp(-{:.2f}*x) + {:.2f}'.format(a,b,c))
            y_hat = func(xdata,a,b,c)
            y_mean = np.mean(ydata)
        
        
            num = np.sum(np.subtract(y_hat, y_mean)**2)
            den = np.sum(np.subtract(ydata, y_mean)**2)
        
            r = np.sqrt(num/den)
            #print(r)
            #axis.text(25, 10, 'r\u00b2: {:.2f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))
            print(S)
            print('r\u00b2: {:.5f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))

ax[1].set_xlabel('Trial Set Number', fontsize=14, fontweight='bold')
        
ax[0].set_xlim([-8,42])
ax[0].set_xticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40])
ax[0].set_xticklabels(['BL']+[str(i) for i in np.arange(0,41,5)], fontsize=12)
ax[1].set_xticklabels(['BL']+[str(i) for i in np.arange(0,41,5)], fontsize=12)

ax[0].set_ylim([0.33, 0.44])
ax[0].set_yticks([0.34, 0.36, 0.38, 0.40, 0.42, 0.44])
ax[0].set_yticklabels([34, 36, 38, 40, 42, 44], fontsize=12)

ax[1].set_ylim([0.49,0.62])
ax[1].set_yticks([0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62])
ax[1].set_yticklabels([50, 52, 54, 56, 58, 60, 62], fontsize=12)




#%%
"""
______________________________________________________________________________

Figure 3B - NEURAL BAR PLOT -  Population Shared Variance
______________________________________________________________________________


"""

  
fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=((7.5,7)))
#fig.suptitle('Population Shared Variance', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Behavioral Timepoint', ha='center', fontweight='bold',fontsize=14)
fig.text(0.0, 0.5, '%sv', va='center', rotation=90, fontweight='bold', fontsize=20)
plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.88,
                    wspace=0.2,
                    hspace=0.1)

fig2, ax2 = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, figsize=((7.5,3)))
#fig.suptitle('Population Shared Variance', fontweight='bold', fontsize=16)
#fig.text(0.5, 0.0, 'Behavioral Timepoint', ha='center', fontweight='bold',fontsize=14)
fig2.text(0.0, 0.5, '%sv', va='center', rotation=90, fontweight='bold', fontsize=16)
plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.88,
                    wspace=0.2,
                    hspace=0.1)
ax2[0].set_xlim([-1,2])

ax[0].set_title(dMonkey['Subject A'], fontsize=16)
ax[1].set_title(dMonkey['Subject B'], fontsize=16)

ax[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax[0].set_yticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=12)

ax2[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax2[0].set_yticklabels([0, 10, 20, 30, 40, 50, 60], fontsize=12)

ind=0
for S, axis in zip(subject_list, [ax[0], ax[1]]):
    axis.set_xlabel('Timepoint', fontweight='bold',fontsize=14)
    
    m50 = np.zeros((4))
    m90 = np.zeros((4))
    s50 = np.zeros((4))
    s90 = np.zeros((4))
    
    mBL = dComp[S]['mBL']
    ind50 = dComp[S]['s_ind50']
    ind90 = dComp[S]['s_ind90']
    
    for i,k in enumerate(['mBL', 'sID', 'sIR', 'sMR']):
 
        r1 = dComp[S][k][ind50]
        r2 = dComp[S][k][ind90]
        
        m50[i] = np.mean(r1)
        m90[i] = np.mean(r2)
        
        s50[i] = stats.sem(r1) 
        s90[i] = stats.sem(r2) 
     
        
    # print(S, '50', stats.ttest_ind(dComp[S]['mBL'][ind50], dComp[S]['sMR'][ind50]))
    # print(S, '90', stats.ttest_ind(dComp[S]['mBL'][ind90], dComp[S]['sMR'][ind90]))
    
    
    dParams = {'50': {'marker':'o', 'color':orange, 'label1': 'Easy (n={})'.format(len(ind50)), 'label2': 'Easy - Baseline'},
               '90': {'marker':'^', 'color':purple, 'label1': 'Hard (n={})'.format(len(ind90)), 'label2': 'Hard - Baseline'}}
                      
    for m, s, degKey, x_axis in zip([m50, m90], [s50, s90], ['50', '90'], [[0,2,4,6], [0.75,2.75,4.75,6.75]]):
        color  = dParams[degKey]['color']
        label1 = dParams[degKey]['label1']
        label2 = dParams[degKey]['label2']
        marker = dParams[degKey]['marker']
    
        axis.bar(x_axis, m, yerr=s, label=label1, color=color, capsize=4, edgecolor='none', alpha=0.8)
        axis.spines[['right', 'top']].set_visible(False)
        

        ax2[ind].bar([0,1], [m[0], m[-1]], yerr=[s[0], s[-1]], width=1, color=color, capsize=4, edgecolor='w', alpha=0.8)
        ax2[ind].spines[['right', 'top']].set_visible(False)
        ax2[ind].set_xticks([0,1])
        ax2[ind].set_xticklabels(['mBL', 'MA'])
        ind+=1
    
    axis.set_xticks([0.5,2.5,4.5,6.5])
    axis.set_xticklabels(['mBL', 'ID', 'IA', 'MA'], fontsize=14)
    #axis.legend(loc='upper right', title='Rotation Condition', ncol=2)
    

'Monkey A Sig. Stars'
lineHeight = 0.475
for x1,x2 in zip([0,2,4,6],[0.75,2.75,4.75,6.75]):
    ax[0].plot([x1, x2],[lineHeight, lineHeight], color='k')

ax[0].text(0,lineHeight*1.02,'n.s.', fontsize=14, fontstyle='italic')
ax[0].text(1.95,lineHeight*0.99,'***', fontsize=20)
ax[0].text(3.95,lineHeight*0.99,'***', fontsize=20)
ax[0].text(5.95,lineHeight*0.99,'***', fontsize=20)

ax2[0].plot([0,1],[lineHeight, lineHeight], color='k')
ax2[0].text(0.2,lineHeight*1.02, 'n.s.', fontsize=14, fontstyle='italic')
ax2[1].plot([0,1],[lineHeight, lineHeight], color='k')
ax2[1].text(0.1,lineHeight*1.0, '***', fontsize=20, fontstyle='italic')

'Monkey B Sig. Stars'
lineHeight = 0.640
for x1,x2 in zip([0,2,4,6],[0.75,2.75,4.75,6.75]):
    ax[1].plot([x1, x2],[lineHeight, lineHeight], color='k')

ax[1].text(0,lineHeight*1.02,'n.s.', fontsize=14, fontstyle='italic')
ax[1].text(2.155,lineHeight*0.99,'*', fontsize=20)
ax[1].text(4.155,lineHeight*0.99,'*', fontsize=20)
ax[1].text(6.1,lineHeight*0.99,'**', fontsize=20)

ax2[2].plot([0,1],[lineHeight, lineHeight], color='k')
ax2[2].text(0.25,lineHeight*1.02, 'n.s.', fontsize=14, fontstyle='italic')

ax2[3].plot([0,1],[lineHeight, lineHeight], color='k')
ax2[3].text(0.25,lineHeight*1.02, 'n.s.', fontsize=14, fontstyle='italic')


ax2[0].set_xlabel('Timepoint', fontsize=14, fontweight='bold')
ax2[1].set_xlabel('Timepoint', fontsize=14, fontweight='bold')
ax2[2].set_xlabel('Timepoint', fontsize=14, fontweight='bold')
ax2[3].set_xlabel('Timepoint', fontsize=14, fontweight='bold')



#%%

"""
______________________________________________________________________________

Figure 4A - Relationship: Behavior + Neural (SHARED)
______________________________________________________________________________


"""

marker1 = 'v'
marker2 = '^'
marker3 = '*'

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(6,8), gridspec_kw={'width_ratios': [1, 1]})
#plt.suptitle('Relationship:\nNeural Variance and Adaptation', fontsize=18, fontweight='bold')
fig.text(0.0, 0.5, 'Adpatation', va='center', fontsize=16, fontweight='bold', rotation = 90)
fig.text(0.5, 0.0, '% change in %sv', ha='center', fontsize=16, fontweight='bold')


plt.subplots_adjust(left=0.12,
                    bottom=0.07,
                    right=0.9,
                    top=0.88,
                    wspace=0.0,
                    hspace=0.15)


for S, axis in zip(subject_list, [ax[0], ax[1]]):
    axis.set_title(dMonkey[S], fontsize=16)
    axis.spines[['right', 'top']].set_visible(False)
    #axis.set_xlabel('% change in %sv', fontsize=16, fontweight='bold')
    sID, sIR, sMR = [dComp[S]['sID'], dComp[S]['sIR'], dComp[S]['sMR']]
    bID, bIR, bMR = [dComp[S]['bID'], dComp[S]['bIR'], dComp[S]['bMR']]
    maxIND = dComp[S]['maxIND']
    
    
    mBL = dComp[S]['mBL']
    ind = dComp[S]['s_ind']
    
    temp1 = []
    temp2 = []
    
    for ind in dComp[S]['s_ind']:
        vSV = dComp[S]['sPE'][ind, :]
        vBE = dComp[S]['rec'][ind, :]
        
        var1 = np.subtract(vSV, mBL[ind])/mBL[ind]
        var2 = vBE
        
        axis.scatter(var1,var2, color='k', alpha=0.1, s=10)
        
        temp1.append(var1)
        temp2.append(var2)
        

    print(stats.pearsonr(np.concatenate((temp1)), np.concatenate((temp2))))
    
    
    X = np.concatenate((temp1)).reshape(-1,1)
    y = np.concatenate((temp2))
    
    print(np.shape(X), np.shape(y))
        
    model = LinearRegression()
    model.fit(X,y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    #print(slope, model.score(X,y))
    
    x1, x2 = [-0.275,0.075]#0.05]
    y1, y2 = [slope*x1+intercept, slope*x2+intercept]
    
    axis.plot([x1, x2],[y1, y2], color='k', zorder=3, lw=2)

ax[0].set_ylim([-3,0.5])
ax[0].set_yticks(np.arange(-3,0.6,0.5))
ax[0].set_yticklabels([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 'BL', 0.5], fontsize=14)

ax[0].axhline(0, color='k', ls='-.', zorder=0)
ax[1].axhline(0, color='k', ls='-.', zorder=0)

ax[0].axvline(0, color='k', ls='-.', zorder=0)
ax[1].axvline(0, color='k', ls='-.', zorder=0)

ax[0].set_xlim([-0.3,0.15])
ax[0].set_xticks([-0.20,  -0.10,  0,  0.10])
ax[0].set_xticklabels([ -20, -10, 'BL',10], fontsize=14)
ax[1].set_xticklabels([ -20, -10, 'BL',10], fontsize=14)


#%%

"""
______________________________________________________________________________

Figure 4B - Relationship: Behavior + Neural (SHARED) - LINE FIT AT KEY POINTS
______________________________________________________________________________


"""
dDeg = {0:'50', 1:'90'}
dSlope = {}

marker1 = '.'
marker2 = '.'
marker3 = '.'

fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(5,7))
#fig.suptitle('Relationship: %sv ~ adaptation', fontsize=14, fontweight='bold')
#fig.text(0.00, 0.0, 'Adaptation', va='center', fontsize=16, fontweight='bold', rotation = 90)#fontsize=15, fontweight='bold')
fig.text(0.50, 0.00, '% change in %sv', ha='center', fontweight='bold', fontsize=16)

plt.subplots_adjust(left=0.075,
                bottom=0.075,
                right=0.9,
                top=1,
                wspace=0,
                hspace=0.1)

for subject_ind, S in enumerate(subject_list):  
    
    dSlope[S] = {}

    indDeg = dComp[S]['s_ind']

    mBL = dComp[S]['mBL']
    sPE = dComp[S]['sPE']
    

    print(S)
    for i, indDeg, marker, ls in zip([0], [indDeg], ['o'], ['--']):    
        dSlope[S][i] = []
        
        sID, sIR, sMR = [dComp[S]['sID'], dComp[S]['sIR'], dComp[S]['sMR']]
        bID, bIR, bMR = [dComp[S]['bID'], dComp[S]['bIR'], dComp[S]['bMR']]


        dTitles = {0:'Initial Drop', 1:'Initial Adaptation', 2:'Max Adaptation'}
        dT = {0: 'ID', 1:'IA', 2:'MA'}
        for j, v1, v2, c, marker in zip([0,1,2], [sID, sIR, sMR],[bID, bIR, bMR], [c1,c2,c3], [marker1, marker2, marker3]):
        
            v1 = np.subtract(v1[indDeg],mBL[indDeg])/mBL[indDeg] 
            v2 = v2[indDeg]
            
            print(dTitles[j])
            slope, intercept, slope_p, LL, UL = sigSlope(v1,v2,0.05)
        
            
            x1, x2 = [np.min(v1),np.max(v1)]
                
            y1 = slope*x1 + intercept
            y2 = slope*x2 + intercept
            ax[j][subject_ind].plot([x1, x2], [y1, y2], color='k', lw=2, ls='-', zorder=2, label='slope: {:.2f}'.format(slope))# ({:.4f})'.format(slope, p))
            
            r,p = stats.pearsonr(v1,v2)
            print('     r: {:.4f}  ({:.4e})'.format(r,p))
            
            if slope_p < 1e-03:
                slope_pstar = '***'
            else:
                slope_pstar = 'ERROR'
            
            ax[j][subject_ind].text(-.275,-2.75, 'slope: {:.2f} ({})'.format(slope,slope_pstar), fontweight='bold', fontsize=13)
            
            ax[j][subject_ind].scatter(v1, v2, marker='o', s=50, edgecolor='k', alpha=0.5, color=c, label='r: {:.2f}'.format(r))#' (***)'.format(r))#label='r: {:.2f} ({:.2e})'.format(r,p))
            dSlope[S][i].append(r)
            
            ax[j][subject_ind].axvline(0, color='k', ls='-.', lw=0.5, zorder=0)

            ax[j][subject_ind].axhline(0, ls='-.', color='k', lw=0.5, zorder=0)
            ax[j][subject_ind].axhline(0, ls='-.', color='k', lw=0.5, zorder=0)
            
            ax[j][subject_ind].spines[['right', 'top']].set_visible(False)

ax[0][0].set_ylabel('Adaptation at ID', fontsize=14, fontweight='bold')
ax[1][0].set_ylabel('Adaptation at IA', fontsize=14, fontweight='bold')
ax[2][0].set_ylabel('Adaptation at MA', fontsize=14, fontweight='bold')

'Max Adaptation'
ax[0][0].set_ylim([-3,0.5])
#ax[0][1].set_ylim([-3,0.5])
ax[0][0].set_yticks(np.arange(-3.0,0.6,0.5))

ylabs = ['{:.1f}'.format(i) for i in np.arange(-3,0.6,0.5)]
ylabs[-2] = 'BL' 

ax[0][0].set_yticklabels(ylabs)


'%change in %sv'
ax[2][0].set_xlim([-0.3,0.15])
ax[2][0].set_xticks([-0.20,  -0.10,  0,  0.10])
ax[2][0].set_xticklabels([ -20, -10, 'BL',10], fontsize=13)
ax[2][1].set_xticklabels([ -20, -10, 'BL',10], fontsize=13)

ax[0][0].set_title('Monkey A', fontsize=16)
ax[0][1].set_title('Monkey B', fontsize=16)




#%%

"""
______________________________________________________________________________

Figure 4C - Relationship: Behavior + Neural (SHARED) - FIT OVER ALL POINTS
______________________________________________________________________________


"""

sigCorr = {'Subject A':[], 'Subject B':[]}

fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False, figsize=(8,3))
#fig.suptitle('Relationship: %sv ~ adaptation', fontsize=14, fontweight='bold')
#fig.text(0.00, 0.0, 'Adaptation', va='center', fontsize=16, fontweight='bold', rotation = 90)#fontsize=15, fontweight='bold')
#fig.text(0.50, 0.00, '% Change in %sv', ha='center', fontweight='bold', fontsize=16)

plt.subplots_adjust(left=0.075,
                bottom=0.075,
                right=0.9,
                top=1,
                wspace=0.0,
                hspace=0.09)

for S, axis in zip(subject_list, [ax[0], ax[1]]):

    ind = dComp[S]['s_ind']
    c = 'k'
    # ind50 = dComp[S]['s_ind50']
    # ind90 = dComp[S]['s_ind90']
    
    #for ind,c in zip([ind50, ind90], [orange, purple]):
    
    LLs = []
    ULs = []
    slopes = []
    
    #for k1, k2 in zip(['sID', 'sIR', 'sMR'],['bID', 'bIR', 'bMR']):
    v1_ = np.array(dComp[S]['sPE'])[ind]
    v2_ = np.array(dComp[S]['rec'])[ind]
    

        
    for k in range(40):
        v1 = v1_[:,k]
        v2 = v2_[:,k]
    
        slope, intercept, p, LL, UL = sigSlope(v1,v2,0.05)
        slopes.append(slope)
        LLs.append(LL)
        ULs.append(UL)
        
        r,p = stats.pearsonr(v1,v2)
        sigCorr[S].append(r)
        
        
    for i in range(40):    
        axis.plot([i,i], [LLs[i], ULs[i]], marker='_', color=c, zorder=0)
        axis.scatter(i,slopes[i], color=c, marker='.', s=50)
        #plt.axhline(slopes[i], ls='-.')
        
    
    def func(x, a, b, c):
        return(a * np.exp(-b*x) + c)
        
    xdata = np.arange(40)
    ydata = slopes
    params, _ = curve_fit(func, xdata, ydata)
    
    a,b,c = params
    
    y_func = func(xdata, a,b,c)
    #axis.plot( np.array(sigCorr[S])*10)
    axis.plot(np.arange(40), y_func, zorder=1, color='k', lw=2, ls='-')
    #axis.text(15, 8, '{:.2f} * exp(-{:.2f}*x) + {:.2f}'.format(a,b,c))
    y_hat = func(xdata,a,b,c)
    y_mean = np.mean(ydata)


    num = np.sum(np.subtract(y_hat, y_mean)**2)
    den = np.sum(np.subtract(ydata, y_mean)**2)

    r = np.sqrt(num/den)
    print(r)
    axis.text(25, 10, 'r\u00b2: {:.2f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))
    print('r\u00b2: {:.2f}  ({:.2f}e(-{:.2f}*x)+{:.2f})'.format(r, a,b,c))

    axis.spines[['right', 'top']].set_visible(False)
    model = LinearRegression()
    model.fit(np.array(xdata).reshape(-1,1),ydata)
    print(model.score(np.array(xdata).reshape(-1,1),ydata))

ax[0].set_ylabel('slope', fontsize=12, fontweight='bold')
ax[1].set_ylabel('slope', fontsize=12, fontweight='bold')

ax[0].text(2,11, dMonkey['Subject A'], fontsize=14)
ax[1].text(2,8, dMonkey['Subject B'], fontsize=14)

ax[1].set_xlabel('Trial Set Number', fontsize=12, fontweight='bold')

ax[0].set_ylim([0,13])
ax[0].set_yticks(np.arange(0,14,2))
ax[1].set_ylim([0,11])
ax[1].set_yticks(np.arange(0,12,2))
# ax[1].set_ylim([0,9])
# ax[1].set_yticks(np.arange(0,10,2))
  

#%%

"""
______________________________________________________________________________

Figure 5 - LDA Classifier - Same Splits for Each Metric within Iteration Number
______________________________________________________________________________

"""
n_folds = 4
n_repeats = 1000



# def cvLDA(Xtrain,ytrain,Xtest,ytest):
    
#     return(model.score(Xtest, ytest), model.score(Xtest,ytest_permuted))


dLDA = {'Subject A': {'model':{'mBL':[], 'sID':[], 'sIR':[], 'sMR':[], 'allTSN':[], 'rateIDIR':[], 'rateIDMR':[]}, 
                      'chance':{'mBL':[], 'sID':[], 'sIR':[], 'sMR':[], 'allTSN':[], 'rateIDIR':[], 'rateIDMR':[]}},
        'Subject B': {'model':{'mBL':[], 'sID':[], 'sIR':[], 'sMR':[], 'allTSN':[], 'rateIDIR':[], 'rateIDMR':[]}, 
                      'chance':{'mBL':[], 'sID':[], 'sIR':[], 'sMR':[], 'allTSN':[], 'rateIDIR':[], 'rateIDMR':[]}}}



for S in subject_list:
    count = 0
    
    ind = dComp[S]['s_ind']
    deg = np.abs(dComp[S]['s_degs'])

    
    for n in range(n_repeats):
        
        X = ind #dComp[S]['mBL'][ind]
        y = deg
        skf = StratifiedKFold(n_splits=n_folds)
        skf.get_n_splits(X, y)
        

        for i, (TN, TT) in enumerate(skf.split(X, y)):

            
            for k in ['mBL', 'sID', 'sIR', 'sMR']:
                
                X = np.array(dComp[S][k][ind]).reshape(-1,1) 
                y = deg
                
                model = LinearDiscriminantAnalysis()
                model.fit(X[TN],y[TN])
                
                dLDA[S]['model'][k].append(model.score(X[TT],y[TT]))
                
                y_permuted = np.random.permutation(y[TT])
                
                dLDA[S]['chance'][k].append(model.score(X[TT],y_permuted))

            
            for k in ['allTSN']:
                
                X = dComp[S]['sPE']
                y = deg
                
                model = LinearDiscriminantAnalysis()
                model.fit(X[TN],y[TN])
                
                dLDA[S]['model'][k].append(model.score(X[TT],y[TT]))
                
                y_permuted = np.random.permutation(y[TT])

                dLDA[S]['chance'][k].append(model.score(X[TT],y_permuted))
                count+=1
                
                
        
            for k in ['rateIDIR']:
                
                X = (dComp[S]['sIR'][ind] - dComp[S]['sID'][ind]).reshape(-1,1)
                y = deg
                
                model = LinearDiscriminantAnalysis()
                model.fit(X[TN], y[TN])
                
                dLDA[S]['model'][k].append(model.score(X[TT], y[TT]))
                
                y_permuted = np.random.permutation(y[TT])
                
                dLDA[S]['chance'][k].append(model.score(X[TT], y_permuted))
            
            
            for k in ['rateIDMR']:
                
                X = (dComp[S]['sMR'][ind] - dComp[S]['sID'][ind]).reshape(-1,1)
                y = deg
                
                model = LinearDiscriminantAnalysis()
                model.fit(X[TN], y[TN])
                
                dLDA[S]['model'][k].append(model.score(X[TT], y[TT]))
                
                y_permuted = np.random.permutation(y[TT])
                
                dLDA[S]['chance'][k].append(model.score(X[TT], y_permuted))
                
        if n%100 == 0:
            print(S, n, n_repeats)
        
print(count)     
            
p_values = []
for S in subject_list:
    print(S)
    for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN', 'rateIDIR', 'rateIDMR']:
        mM, mC = np.mean(dLDA[S]['model'][k]), np.mean(dLDA[S]['chance'][k])
        
        'p-value: what fraction of chance is greater than the mean model performance?'


        v1 = np.mean(dLDA[S]['model'][k])
        v2 = dLDA[S]['chance'][k]

        count = 0
        for i in v2:
            
            if np.abs(i) > v1:
                count+=1
        
        print('\t{} - {:.3f}% ({:.3f}) (p={:.2e})'.format(k, mM*100, mC*100, count/len(v2)))
        p_values.append(count/len(v2))
    print('\n')


#%%
'Pairwise comparisons of means of models'

for S in subject_list:
    print(S)

    model_category = np.concatenate(([ [k]*4000 for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]))
    model_accuracy = np.concatenate(([ np.array(dLDA[S]['model'][k])*100 for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]))
    
    
    df = pd.DataFrame({'model': model_category,
                       'accuracy': model_accuracy})
    
    
    model = ols('accuracy ~ C(model)', data=df).fit()
    
    print('\t', sm.stats.anova_lm(model, typ=1))
    

    tukey = pairwise_tukeyhsd(endog=df['accuracy'],
                              groups=df['model'],
                              alpha=0.05)
    

    print(tukey)




#%%


"""
______________________________________________________________________________

FIGURE 5A: CLASSIFIER (LDA) BAR PLOT - BL, ID, IA, MA, allTSN
______________________________________________________________________________

"""

width = 0.75
x_loc = [0, width, width*2, width*3, width*4]

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(3,3))
fig.text(0.45,0, 'Timepoint', fontsize=10, fontweight='bold', ha='center')
plt.subplots_adjust(left=0.0,
                bottom=0.11,
                right=0.9,
                top=1,
                wspace=0.0,
                hspace=0.0)

for S, axis in zip(subject_list, [ax[0], ax[1]]):
    m = [ np.mean(dLDA[S]['model'][k])  for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]
    e = [ stats.sem( dLDA[S]['model'][k] )  for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]
    
    axis.bar(x_loc, m, yerr=e, width=width, color=['darkgrey', magenta, yellow, blue, 'lightgrey'], edgecolor='w', alpha=0.9)
    
    mRC = [ np.mean(dLDA[S]['chance'][k])  for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]
    eRC = [ stats.sem(dLDA[S]['chance'][k])  for k in ['mBL', 'sID', 'sIR', 'sMR', 'allTSN']]
    
    axis.errorbar(x_loc, mRC, yerr=eRC, linestyle='', marker='_', markersize=12, zorder=10, color='k')
    
    axis.set_xlim([-width,x_loc[-1]+width])
    axis.set_xticks([])
    axis.set_xticks(x_loc)
    axis.set_xticklabels(['BL', 'ID', 'IA', 'MA', 'TSN'], fontsize=8, rotation=0)
    
    axis.set_yticks(np.arange(0,1.1,0.1))
    axis.set_yticklabels(np.arange(0,101,10), fontsize=8)
    
    axis.set_title(dMonkey[S], fontsize=10)
    axis.spines[['right', 'top']].set_visible(False)
    

ax[0].set_ylabel('Mean Accuracy (%)', fontweight='bold', fontsize=10)

"""ADD P VALUES"""

ax[0].text(x_loc[0], 0.60, 'n.s.', ha='center', fontstyle='italic')
ax[0].text(x_loc[1], 0.82, '*', ha='center', fontstyle='italic')
ax[0].text(x_loc[2], 0.82, '**', ha='center', fontstyle='italic')
ax[0].text(x_loc[3], 0.82, '*', ha='center', fontstyle='italic')
ax[0].text(x_loc[4], 0.60, 'n.s.', ha='center', fontstyle='italic')

ax[1].text(x_loc[0], 0.55, 'n.s.', ha='center', fontstyle='italic')
ax[1].text(x_loc[1], 0.72, 't.', ha='center', fontstyle='italic')
ax[1].text(x_loc[2], 0.72, 't.', ha='center', fontstyle='italic')
ax[1].text(x_loc[3], 0.62, 'n.s.', ha='center', fontstyle='italic')
ax[1].text(x_loc[4], 0.55, 'n.s.', ha='center', fontstyle='italic')



#%%
"""
______________________________________________________________________________

FIGURE 5B: CLASSIFIER (LDA) - EACH TSN
______________________________________________________________________________

"""

dLDA_eachTSN = {'Subject A': {'model':{}, 'chance':{}},
        'Subject B': {'model':{}, 'chance':{}}}

'Compute model and chance accuracy.'
for S in subject_list:

    ind = dComp[S]['s_ind']
    sPE = dComp[S]['sPE'][ind,:]
    deg = np.abs(dComp[S]['s_degs'])
    
    skf = StratifiedKFold(n_splits=4)
    
        
    dLDA_eachTSN[S]['model'] = np.zeros((40, 1000, 4))
    dLDA_eachTSN[S]['chance'] = np.zeros((40, 1000, 4))
        
    count=0
    for n in range(1000):
        skf.get_n_splits(ind, deg)

        for i, (TN, TT) in enumerate(skf.split(ind, deg)):
        
        
            for k in np.arange(40):
                
                X = (sPE[:,k]).reshape(-1,1)
                y = deg
                
                model = LinearDiscriminantAnalysis()
                model.fit(X[TN], y[TN])
                
                dLDA_eachTSN[S]['model'][k,n,i] = model.score(X[TT], y[TT])
                
                y_permuted = np.random.permutation(y[TT])
                
                dLDA_eachTSN[S]['chance'][k,n,i] = model.score(X[TT], y_permuted)
         
                count+=1
        if (k==39) and (n%10 == 0):
            print(S, n, 1000)
        #print(S, k, np.mean(dLDA_eachTSN[S]['model'][k]), np.mean(dLDA_eachTSN[S]['chance'][k]))

    
#%%
sig = {'Subject A':[], 'Subject B':[]}

'Compute P-Value (compared to chance).'
for S in subject_list:
    for k in range(40):

        v1 = np.mean(dLDA_eachTSN[S]['model'][k,:,:])
        v2 = np.concatenate((dLDA_eachTSN[S]['chance'][k,:,:]))

        count = 0
        for i in v2:
            if np.abs(i) > v1:
                count+=1
        
        sig[S].append(count/len(v2))
        

#%%
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
obj_to_pickle = [dLDA, dLDA_eachTSN]
filename = 'LDA_4-fold_1e3.pkl' #'LDA_4-fold-1e4.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()

#%%
"""
______________________________________________________________________________

FIGURE 5C: CLASSIFIER (LDA) - EACH TSN - LINE PLOT
______________________________________________________________________________

"""

open_file = open('LDA_4-fold_1e3.pkl', "rb") 
loaded = pickle.load(open_file)
_, dLDA_all = loaded
open_file.close()

fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False, figsize=((10,8)))
#fig.suptitle('LDA Model Accuracy - Trial Sets', fontweight='bold', fontsize=16)
fig.text(0, 0.5,'Mean Accuracy (%)', va='center', fontweight='bold', fontsize=20, rotation=90)

plt.subplots_adjust(left=0.075,
                    bottom=0.1,
                    right=0.9,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.15)


ax[1].set_xlabel('Trial Set Number', fontsize=20, fontweight='bold')


for S, axis in zip(subject_list, [ax[0], ax[1]]):

    
    meanModel = np.array([np.mean(dLDA_all[S]['model'][k]) for k in range(40)])
    semModel  = np.array([stats.sem(dLDA_all[S]['model'][k]) for k in range(40)])
    
    meanRC =  np.mean([np.mean(dLDA_all[S]['chance'][k]) for k in range(40)])
    
    print(S)
    m, b, p, LL, UL = sigSlope(np.arange(40),meanModel,0.05)
    print('     slope: {:.4f} {:.4f}'.format(m*100, p))# LL, UL)

    x1, x2 = [0,40]
    y1, y2 = [m*x1+b, m*x2+b]
    axis.plot([x1,x2], [y1,y2], color='k', ls='-', lw=2, zorder=0)
    r,p = stats.pearsonr(np.arange(40), meanModel)

    axis.scatter(0, meanModel[0], color=magenta, marker='v', s=400, zorder=0, alpha=1)
    axis.scatter(1, meanModel[1], color=yellow, marker='^', s=400, zorder=0, alpha=1)
    axis.scatter(np.arange(40), meanModel, color='k', marker='o', edgecolor='w', zorder=3, label=dMonkey[S], s=100)
    

    ymin = np.min(meanModel)-0.05
    ymax = np.max(meanModel)+0.05
    axis.set_ylim([ymin,ymax])
    
    #print(np.max(meanModel) - np.min(meanModel))
    
    
    if S == 'Subject A':
        #axis.set_ylim([0.61,0.875])
        axis.set_yticks(np.arange(0.65,0.86,0.05))
        axis.set_yticklabels([65, 70, 75, 80, 85], fontsize=14)
    else:
        # axis.set_ylim([0.47,0.77])
        axis.set_yticks(np.arange(0.50,0.76,0.05))
        axis.set_yticklabels([50, 55, 60, 65, 70, 75], fontsize=14)

    axis.text(-1,ymax+0.01,dMonkey[S], fontweight='bold', fontsize=20)
    axis.spines[['right', 'top']].set_visible(False)

ax[1].set_xlim([-1,41])
ax[1].set_xticks(np.arange(0,41,5))
ax[1].set_xticklabels(np.arange(0,41,5), fontsize=15)

#%%


"""
______________________________________________________________________________

FIGURE 5C: PREDICTION (LINEAR REGRESSION) - IR - TRIAL SET NUMBER OF MAX ADAPTATION

Note: Only IR (of BL, ID, IR, MR) is predictive of TSN
______________________________________________________________________________

"""

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=True, figsize=((6,3)))
#fig.suptitle('Relationship of %sv and TSN', fontweight='bold')
#fig.text(0.5, 0, '%sv (Initial Recovery)', fontweight='bold', ha='center', fontsize=12)
fig.text(0,0.5, 'Trial Set Number', fontweight='bold', va='center', rotation=90, fontsize=10)

plt.subplots_adjust(left=0.075,
                    bottom=0.125,
                    right=1,
                    top=0.80,
                    wspace=0.0,
                    hspace=0.15)

ax[0].set_title('Monkey A', fontsize=10)
#ax[0].set_xlim([0.27, 0.47])
ax[0].set_xticks(np.arange(0.28,0.48,0.02))
ax[0].set_xticklabels(['{:.0f}'.format(i*100) for i in np.arange(0.28,0.48,0.02)], fontsize=8)
ax[0].set_ylim([-1,42])
ax[0].set_yticks(np.arange(0,41,5))
ax[0].set_yticklabels(np.arange(0,41,5), fontsize=8)
ax[0].set_xlabel('%sv', fontweight='bold', fontsize=10)


ax[1].set_title('Monkey B', fontsize=10)
ax[1].set_xticks(np.arange(0.46,0.65,0.02))
ax[1].set_xticklabels(['{:.0f}'.format(i*100) for i in np.arange(0.46,0.65,0.02)], fontsize=8)
ax[1].set_xlabel('%sv', fontweight='bold', fontsize=10)


for S, axis in zip(subject_list, [ax[0], ax[1]]):
    
    mBL = dComp[S]['mBL']
    ind = dComp[S]['s_ind']
    
    
    for k in ['sIR']:#['mBL', 'sID', 'sIR', 'sMR']:
        X = dComp[S][k][ind]
        y = dComp[S]['maxIND'][ind]

        print(S, k)

        
        m, b, p, LL, UL = sigSlope(X,y,0.05)
        print('\tslope: {:.4f} ({:.4e})  ({:.4f},{:.4f})'.format(m,p,LL,UL))
     

        if S == 'Subject A':
            x1, x2 = [0.27,0.47]
        else:
            x1, x2 = [0.45,0.64]
            
            
        y1, y2 = [m*x1+b, m*x2+b]
        
        axis.plot([x1,x2], [y1,y2], color='k', ls='-.')
        axis.scatter(X, y, color=yellow, edgecolor='k', label=dMonkey[S])
        axis.spines[['right', 'top']].set_visible(False)

        r, p = stats.pearsonr(X, y) 
        print('     r: {:.2f} ({:.3f})'.format(r,p))



#%%
"""
______________________________________________________________________________

FIGURE 5: PREDICTION (LINEAR REGRESSION) - ID, IR, MR - MAX ADAPTATION

NOTE: BL is NOT significantly predictive of max adaptation.
______________________________________________________________________________

"""

slopes = {'Subject A':[], 'Subject B':[]}
CI = {'Subject A': {'LL':[], 'UL':[]},
      'Subject B': {'LL':[], 'UL':[]}}

fig, ax = plt.subplots(ncols=2, nrows=3, sharex=False, sharey=True, figsize=((5,3.5*3)))

plt.subplots_adjust(left=0.09,
                    bottom=0.125,
                    right=1,
                    top=0.80,
                    wspace=0.1,
                    hspace=0.25)

ax[0][0].set_title('Monkey A', fontsize=16)
ax[0][1].set_title('Monkey B', fontsize=16)

Amin, Amax = [0.27, 0.49]
Bmin, Bmax = [0.41, 0.68]

for i in range(3):
    ax[i][0].set_xlim([Amin, Amax])
    ax[i][0].set_xticks(np.arange(Amin+0.01,Amax,0.04))
    ax[i][0].set_xticklabels(['{:.0f}'.format(i*100) for i in np.arange(Amin+0.01,Amax,0.04)])
    ax[i][0].set_ylabel('Max Adaptation', fontsize=14, fontweight='bold')
    ax[i][0].spines[['right', 'top']].set_visible(False)
    

    
    ax[i][1].set_xlim([Bmin, Bmax])
    ax[i][1].set_xticks(np.arange(Bmin+0.01,Bmax,0.04))
    ax[i][1].set_xticklabels(['{:.0f}'.format(i*100) for i in np.arange(Bmin+0.01,Bmax,0.04)])
    ax[i][1].spines[['right', 'top']].set_visible(False)
    
ax[2][0].set_yticks(np.arange(-0.4,0.3,0.2))
ax[2][0].set_yticklabels([-0.4, -0.2, 'BL', 0.2], fontsize=12)


axA = [ax[0][0], ax[1][0], ax[2][0]]
axB = [ax[0][1], ax[1][1], ax[2][1]]

for S, axis_list in zip(subject_list, [axA, axB]):

    
    for k, axis, c in zip(['sID', 'sIR', 'sMR'], axis_list, [magenta, yellow, blue]):

        ind = dComp[S]['s_ind']

           
        X = np.array(dComp[S][k][ind]).reshape((-1,1))
        y = np.array(dComp[S]['bMR'][ind])
        
        m, b, p, LL, UL = sigSlope(X,y,0.05)
        r,p = stats.pearsonr(dComp[S][k][ind], dComp[S]['bMR'][ind])
        print('\t\t {}, r: {:.4f} ({:.2e})'.format(k,r,p))
        
        slopes[S].append(m)
        CI[S]['LL'].append(LL)
        CI[S]['UL'].append(UL)
        
        r,p = stats.pearsonr(X[:,0],y)     
        
        if p < 1e-03:
            rstar = '***'
        elif p < 1e-02:
            rstar = '**'
        elif p < 0.05:
            rstar = '*'
        elif (p > 0.05) and (p < 0.1):
            rstar = 't.'
        else:
            rstar = 'n.s.'
        
        if S == 'Subject A':
            axis.text(.29,0.1, 'r: {:.2f} ({})'.format(r,rstar), fontsize=13, fontweight='bold')
        else:
            axis.text(.43,0.1, 'r: {:.2f} ({})'.format(r,rstar), fontsize=13, fontweight='bold')
            print(p, rstar)

        if S == 'Subject A':
            x1, x2 = [Amin,Amax]
        else:
            x1, x2 = [Bmin,Bmax]
        
        y1, y2 = [m*x1+b, m*x2+b]
        
        
        axis.axhline(0, color='lightgrey', zorder=0)
        axis.plot([x1,x2], [y1,y2], color='k', ls='-.')
        axis.scatter(X[:,0], y, color=c, edgecolor='k', label=dMonkey[S])


ax[0][0].set_xlabel('%sv at ID Timepoint', fontweight='bold', fontsize=12)
ax[0][1].set_xlabel('%sv at ID Timepoint', fontweight='bold', fontsize=12)

ax[1][0].set_xlabel('%sv at IA Timepoint', fontweight='bold', fontsize=12)
ax[1][1].set_xlabel('%sv at IA Timepoint', fontweight='bold', fontsize=12)

ax[2][0].set_xlabel('%sv at MA Timepoint', fontweight='bold', fontsize=12)
ax[2][1].set_xlabel('%sv at MA Timepoint', fontweight='bold', fontsize=12)
