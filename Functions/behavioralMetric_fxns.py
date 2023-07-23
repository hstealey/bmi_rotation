# -*- coding: utf-8 -*-
"""

This script is designed to compute the behavioral adaptation metric.


Pre-requisites:
        Files:        {subject}_FA_loadings_40_sets.pkl 
                      
        From Script:  generatepickles.py
        
        
Functions:
    



'Projection'
'https://flexbooks.ck12.org/cbook/ck-12-college-precalculus/section/9.6/primary/lesson/scalar-and-vector-projections-c-precalc/'



Created on Thu Dec  1 20:16:47 2022
Last updated April 23, 2023
@author: hanna
"""

import os
import glob 
import numpy as np
#from scipy import stats
import pandas as pd
#from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
#import seaborn as sb
#from matplotlib.lines import Line2D
#from bioinfokit.analys import stat
#from numpy.linalg import norm 
#from collections import Counter


#import pickle
# os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')

# fn = glob.glob('*_FA_loadings_40sets_noTC.pkl')
# open_file = open(fn[0], "rb")
# Aev = pickle.load(open_file)
# open_file.close()






#%%

"""
_______________________________________________________________
_______________________________________________________________

FUNCTION INPUT

    Factor Loadings & Behavior (trial times, distances) 
    ___________________________________________________    
    
    File Name:    {subject}_FA_loadings_40sets_noTC.pkl 
    From Script:  Functions/generatePickles.py
    
    Parameters Saved:
            [0]      [1]   [2]    [3]    [4]     [5]
    Sev = deg_list, dates, dEV, dTimes, dDist, numUnits
    
    
________________________________________________________________
"""
def getAdaptation(Sev, returnMag=False):

    """
    '################################################################'
    'Computing Behavioral Adaptation over Sets of 8 Consecutive Trials'
    '################################################################'
    
    INPUT
        Sev: pickle (see above)
        
    OUTPUTS

        adt: adaptation for each set of trials
             adt[session#] (dictionary of lists)
        
        dID: initial deficit caused by the rotation (first set of trials in perturbation)
             dID  (list)
             
             
        dIA: initial adaptation after rotation (second set of trials in perturbation)
             dIA  (list)
             
        dMA: maximum adaptation after rotation
             dMA  (list)
    
             
    """
    
    
    #         [0]      [1]   [2]    [3]    [4]     [5]
    # Sev = deg_list, dates, dEV, dTimes, dDist, numUnits
    adt  = {}
    dID  = []
    dIA  = []
    dMA  = []
    iMA  = []
    
    mag =  []

    numSessions = len(Sev[0])
    
    for i, ind in enumerate(Sev[1]):
        adt[i] = []
        
        tBL = np.mean(Sev[3][i]['BL'], axis=0)
        tPE = np.mean(Sev[3][i]['PE'], axis=0)
        
        dBL = np.mean(Sev[4][i]['BL'], axis=0) 
        dPE = np.mean(Sev[4][i]['PE'], axis=0) 
        
       # mag.append(np.sqrt(tPE**2+dPE**2))
        
        mT = np.mean(tBL)
        mD = np.mean(dBL)
        
        x = np.subtract(np.divide(tPE, mT),1)
        y = np.subtract(np.divide(dPE, mD),1)
        
        
        xy = np.sqrt(x**2 + y**2)
        xy_ind = np.where(xy==np.min(xy))[0][0]
        
        dInit = xy[0] 
        #dAct = xy[xy_ind]
    
        for j in range(len(x)):
            #proj = np.dot([x[0],y[0]],[x[j], y[j]])/dInit   
            #rec[i].append((dInit - proj)/dInit)
            #rec[i].append(-1*np.dot([x[0],y[0]],[x[j], y[j]]))
            
            len_proj = -1*np.dot([x[j], y[j]],[x[0],y[0]])/xy[0]
            adt[i].append(len_proj)
            
            
            
            """
            Adaptation (rec) is defined as the LENGTH of the line projected onto BL-ID.
                To represent the fact that increased distances below baseline are undesireable, I multiplied the value by -1.
                Values below 0 (BL) represent sets with increased time and/or distance from baseline.
            
            a * b / ||b||
            
            """

        
        iMA.append(np.where(np.array(adt[i]) == np.max(adt[i]))[0][0])
        
        dID.append(adt[i][0])
        dIA.append(adt[i][1])
        dMA.append(adt[i][iMA[-1]])
        
        
        # 'Example Plot'
        # 'IBM'
        # blue    = [100/255, 143/255, 255/255]
        # purple  = [120/255, 94/255, 240/255]
        # magenta = [220/255, 38/255, 127/255]
        # orange  = [254/255, 97/255, 0/255]
        # yellow  = [255/255, 176/255, 0/255]

        
        # if i == 25: #Subject B

        
        #     BLx = mT/mT - 1
        #     BLy = mD/mD - 1
            
        #     IDx = -1*(tPE[0]/mT - 1)
        #     IDy = -1*(dPE[0]/mD - 1)
            
        #     IAx = -1*(tPE[1]/mT - 1)
        #     IAy = -1*(dPE[1]/mD - 1)
            
        #     MAx = -1*(tPE[iMA[-1]]/mT - 1)
        #     MAy = -1*(dPE[iMA[-1]]/mD - 1)
            
        #     plt.figure(figsize=((3,3)))
            
        #     ind = np.arange(40)
        #     ind = np.delete(ind, [0,1,iMA[-1]])
            
        #     plt.scatter(-1*(tPE[ind]/mT -1), -1*(dPE[ind]/mD - 1), color='white', edgecolor='k', alpha=0.25, s=10,zorder=0)
            
        #     plt.scatter(BLx, BLy, color='k', s=50, edgecolor='k')
        #     plt.scatter(IDx, IDy, color=magenta, s=50, edgecolor='k')
        #     plt.scatter(IAx, IAy, color=yellow, s=30, edgecolor='k')
        #     plt.scatter(MAx, MAy, color=blue, s=30, edgecolor='k')
            
        #     'Baseline-Initial Drop'
        #     plt.plot([BLx, IDx],[BLy, IDy], color=magenta, zorder=0)
            
        #     'Baseline-Initial Adaptation'
        #     plt.plot([BLx, IAx],[BLy, IAy], color=yellow, zorder=0)#, ls='-.')
            
        #     'Baseline-Max Adaptation'
        #     plt.plot([BLx, MAx],[BLy, MAy], color=blue, zorder=0)#, ls='-.')
            
            
        #     'SLOPE OF BASELINE-INTIAL DROP'
        #     slope_ID  = IDy/IDx  
        #     intercept_ID = IDy - slope_ID*IDx
            
        #     slope_perpID = -1/slope_ID
            
        #     'PERP - Baseline-Intial Adaptation'
        #     intercept_perpID_atIA  = IAy - slope_perpID*IAx

        #     xproj_IA = (intercept_perpID_atIA - intercept_ID)/(slope_ID - slope_perpID)
        #     yproj_IA = slope_perpID*xproj_IA + intercept_perpID_atIA
            
            
        #     'PERP - Baseline-Max'     
        #     intercept_perpID_atMA = MAy - slope_perpID*MAx
            
        #     xproj_MA = (intercept_perpID_atMA - intercept_ID)/(slope_ID - slope_perpID)
        #     yproj_MA = slope_perpID*xproj_MA + intercept_perpID_atMA
            
            
        #     'Plot projections.'
        #     plt.scatter(xproj_IA, yproj_IA, color=yellow, s=100, marker='*', edgecolor='k')
        #     plt.plot([IAx, xproj_IA], [IAy, yproj_IA], color='grey', zorder=0, ls='-')
            
        #     plt.scatter(xproj_MA, yproj_MA, color=blue, s=100, marker='*', edgecolor='k')
        #     plt.plot([MAx, xproj_MA], [MAy, yproj_MA], color='grey', zorder=0, ls='-')

        #     plt.axis('equal')

        #     lenID = dID[-1]
        #     lenIR = np.sqrt(xproj_IA**2+yproj_IA**2)
        #     lenMR = np.sqrt(xproj_MA**2+yproj_MA**2)
            
        #     plt.axhline(0, xmax=0.9, color='lightgrey', ls='-.', zorder=0)
        #     plt.axvline(0, ymax=0.97, color='lightgrey', ls='-.', zorder=0)
            
            
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.show()
            
            

    
    dDegs = {'50':50, 'neg50':-50, '90':90, 'neg90':-90}
    degs   = [dDegs[deg] for deg in Sev[0]]
    dfAdt = pd.DataFrame({'deg4':degs , 'deg': np.abs(degs), 'ID': dID, 'IR': dIA, 'MR': dMA, 'indMR': iMA})
    if returnMag==True:
        return(pd.DataFrame({'deg': np.abs(degs), 'mag': mag}))
    else:
        return(adt, dfAdt)
    
                


#adt, df = getAdaptation(Aev, returnMag=False)

#%%

'Matching Example Barplot'

# i = 25

# 'IBM'
# blue    = [100/255, 143/255, 255/255]
# purple  = [120/255, 94/255, 240/255]
# magenta = [220/255, 38/255, 127/255]
# orange  = [254/255, 97/255, 0/255]
# yellow  = [255/255, 176/255, 0/255]

# lenID = df['ID'][i]
# lenIR = df['IR'][i]
# lenMR = df['MR'][i]

# plt.figure(figsize=(2,4))
# plt.bar(0, lenID, width=1.0, color=magenta, edgecolor='k')
# plt.bar(1, lenIR, width=1.0, color=yellow, edgecolor='k')
# plt.bar(2, lenMR, width=1.0, color=blue, edgecolor='k')

# plt.xlim([-0.75,2.75])
# plt.xticks([])
# #plt.xticks([0,1,2], ['Initial Drop', 'Intial Adaptation', 'Max Adaptation'], fontname='Arial', fontsize=10, fontweight='bold')
# plt.yticks(np.arange(-0.7, 0.1, 0.1), [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 'BL'], fontname='Arial', fontsize=10)
# plt.ylabel('Adaptation', fontname='Arial', fontsize=12, fontweight='bold')







#%%
def behaviorBL(Sev):
   

    """
    '################################################################'
    'Calculating BASELINE Behavoir over Sets of 8 Consecutive Trials'
    '################################################################'
    
    INPUT
        S: pickle (see above)
        
    OUTPUTS

        behavior
    
             
    """
    
    
  #        [0]   [1]   [2]   [3]     [4]     [5]
  # Sev = degs, dates, dEV, dTimes, dDist, numUnits
  
    behavior  = {}

    numSessions = len(Sev[0])
    
    for i, ind in enumerate(Sev[1]):
        
        behavior[i] = {}
        tBL = np.mean(Sev[3][i]['BL'], axis=0)
        dBL = np.mean(Sev[4][i]['BL'], axis=0) 
        
        behavior[i]['time'] = tBL
        behavior[i]['dist'] = dBL
    
    return(behavior)
    



