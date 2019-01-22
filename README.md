# ton_rfmri_repo

Scripts used for analysis of resting state fMRI data in premanifest Huntington's Disease in:
*"Resting-state connectivity stratifies premanifest Huntington's disease by cognitive decline rate"*
Authors: Pablo Polosecki, Eduardo Castro, Irina Rish, Dorian Pustina, John H. Warner, Andrew Wood, Cristina Sampaio and Guillermo A. Cecchi

This is NOT a toolbox. The scripts are provided "as is" for the purposes of reproducibility and transparency. If interested in adapting these scripts, feel free to  contact Pablo Polosecki (pipolose@us.ibm.com) for assistance.


Most scripts are written in *Python (2.7)*, except for the ones in **fcd_tools**, which are written in *MATLAB*
This repository contains the following folders (each folder has its own README file with a description of its contents):

* **explore**: Scripts for manipulating files from the raw dataset, and obtaining longitudinal slopes of cognitive change.

* **preprocess**: Scripts for preprocessing fMRI time series and performing registration to the MNI template.

* **fcd_tools**: Scripts for computing FCD feature maps.

* **polyML**: Scripts used for cross-validated classifications.

* **my_nipype_io**: Modification of a nipype's io interface to deal with modernly formatted strings.

* **plotting_tools**: Functions for plotting tasteful statistical maps on MNI space.

* **figure_scripts**: Scripts used for producing brain map figures in the paper.

**test_retest_correlations**: Ipython notebook computing similarity between test and retest FCD maps.

* **controls**: Scripts for controling for confounds reported in the paper.

   **controls/combat_harmonization**: Corrects FCD features by site using combat

   **controls/combat_harmonization**: Corrects FCD features by site using combat.

   **controls/VBM_controls**: corrects FCD features by gray matter concentration. Assumes VBM maps exist. (In our case, these were provided by Castro et a. 2018)

   **controls/n_visit_controls**: controls for effect of number of visits in assigment to extreme subgroups of decline

   **controls/site_effects**: Checks for differences in demographics/cognition across sites.

   **controls/exclusion_numbers**: Ipython notebook computing how many subjects where excluded for what reason


**Dependencies**
The following softwares and libraries are required for running the scripts:
Python 2.7, Pandas, Numpy, Matplotlib, Seaborn, Scickit-learn, Scikit-contrib Lightning, Nilearn, Nipy, Nipype, FSL, FreeSurfer, MATLAB

**References**
Castro, E., Polosecki, P., Rish, I., Pustina, D., Warner, J. H., Wood, A., et al. (2018). Baseline multimodal information predicts future motor impairment in premanifest Huntington's disease. NeuroImage: Clinical, 19, 443?453. http://doi.org/10.1016/j.nicl.2018.05.008
