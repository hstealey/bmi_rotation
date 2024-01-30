# bmi_rotation


Usage Information

This code is used to analyze brain-machine interface (BMI) data from a visusomotor rotation task.  It is intended to go along with the following manuscript: Neural population variance explained adaptation differences during learning.

For task code, see https://github.com/santacruzlab  (branch: bmi_ssd) bmi_python>built_in_tasks>error_clamp_tasks.py: class BMICursorVisRotErrorClamp.



# Main Scripts
**main_plots.py**
	This script is designed to visually explore (i.e., plot) the relationship of neural spiking variance (as determined by factor analysis) and behavior (adaptation to a rotation perturbation).  It contains the code figures 3-6 of the manuscript. Note: Nomenclature in the scrip and the manuscript may vary (initial drop/ initial deficit - ID; initial adaptation/ initial recovery - IA/IR; maximum adaptation/maximum recovery - MR/MA).

	Figure 3: Behavioral adaptation
	Figure 4: Neural (Shared Variance (%sv)) changes
	Figure 5: Relationship between behavior and %sv changes
	Figure 6: Prediction of MA using timepoint %sv changes

	Pre-requisites:
		Decoder unit factor loadings to compute variance components; Behavior (cursor distance, trial times)
		File Name (e.g.):    subject+'*_FA_loadings_40sets_alt_LOOCV-90_experimental90Var_20240124.pkl''
        From Script:  generatePickles.py          
                [0]       [1]    [2]  [3]      [4]       [5]     [6]      [7]
          Sev = deg_list, dates, dEV, dTimes_, dDist_, numUnits, numBins, sc_preZ




**main_stats.py**
	This script is used to compute statistics (F-test, t-test, ANOVA + post-hoc Tukey Kramer)

	Test for Differences in Rotation Condition
		2x two-way ANOVA to test if [1] +/-50 and [2] +/-90 produce significant differences in behavior at each timepoint
	Test for Difference in Number of Decoder Units 
		F-test + t-test 

	Behavioral
		F-test + t-test for adaptation at each timepoint

	DEPRECATED:
	Neural
		two-way ANOVA + post-hoc testing for %sv at each timepoint


**supplementary_plots.py** + **supplementary.py**
	These scripts are used to plot the supplementary figures.



# Support Functions (custom)

**behavioralMetric_fxns.py**
	The purpose of this script is to calculate the ADAPTATION metric as well as pull baseline metrics (time, distance) for each trial.

**behavior_successRate.py**
	This script analyzes what percentage of trials were successful on the first or second attempt.

**getDataDictionary_fxn.py**
	This script contains a function that formats the process data (.pkl) to be used to plot (main_plot.py) and perform additional stats (main_stats.py).

**stats_fxns.py**
	This script contains a custom written F-test as well as a test to determine if a linear regression slope is significantly different from 0.


The following files are used to format and process data (e.g., choosing the number of factors including cross-validation, factor analysis) :  **generatePickles.py**, **generatePickles_fxns.py**








	



#Updates
01.30.2024: post-review code revisions