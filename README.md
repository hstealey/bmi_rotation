# bmi_rotation


Usage Information

This code is used to analyze brain-machine interface (BMI) data from a visusomotor rotation task.  

For task code, see https://github.com/santacruzlab  (branch: bmi_ssd) bmi_python>built_in_tasks>error_clamp_tasks.py: class BMICursorVisRotErrorClamp.


# Main Scripts
**main_plots.py**
	This script is designed to visually explore (i.e., plot) the relationship of neural spiking variance (as determined by factor analysis) and behavior (adaptation to a rotation perturbation).  It contains the code figures 2-5 of the manuscript. Note: Nomenclature in the scrip and the manuscript may vary (initial drop/ initial deficit - ID; initial adaptation/ initial recovery - IA/IR; maximum adaptation/maximum recovery - MR/MA).

	Figure 2: Behavioral adaptation
	Figure 3: Neural (Shared Variance (%sv)) changes
	Figure 4: Relationship between behavior and neural changes
	Figure 5: Classification of "easy" versus "hard" rotation conditions

	Pre-requisites:
		Decoder unit factor loadings to compute variance components; Behavior (cursor distance, trial times)
		File Name:   subject+_FA_loadings_40sets.pkl'
		From Script: Neural-And-Behavior-Relationships.py 
					Sev = degs, dates, dTC, dEV, dTimes, dDist, subject




**main_stats.py**
	This script is used to compute statistics (F-test, t-test, ANOVA + post-hoc Tukey Kramer)

	Test for Differences in Rotation Condition
		2x two-way ANOVA to test if [1] +/-50 and [2] +/-90 produce significant differences in behavior at each timepoint
	Test for Difference in Number of Decoder Units 
		F-test + t-test 
	Behavioral
		F-test + t-test for adaptation at each timepoint
	Neural
		two-way ANOVA + post-hoc testing for %sv at each timepoint

**supplementary.py**


**supplementary_plots.py**



# Support Functions (custom)

**behavioralMetric_fxns.py**
	The purpose of this script is to calculate the ADAPTATION metric as well as pull baseline metrics (time, distance) for each trial.


**stats_fxns.py**
	This script contains a custom written F-test as well as a test to determine if a linear regression slope is significantly different from 0.



The following files are used to process and format data (.hdf5): 
**generatePickles.py**

**generatePickles_fxns.py**





	



#Updates
07.23.2023: intial commit for manuscript 1