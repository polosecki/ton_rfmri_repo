# ton_rfmri_repo

Scripts used for analysis of resting state fMRI data in premanifest Huntington's Disease in:
*"Resting-state connectivity stratifies premanifest Huntington's disease by cognitive decline rate"*
Authors: Pablo Polosecki, Eduardo Castro, Irina Rish, Dorian Pustina, John H. Warner, Andrew Wood, Cristina Sampaio and Guillermo A. Cecchi

Most scripts are written in *Python (2.7)*, except for the ones in **fcd_tools**, which are written in *MATLAB*
This repository contains the following folders (each folder has its own README file with a description of its contents):

* **explore**: These are scripts for manipulating files from the raw dataset, and obtaining longitudinal slopes of cognitive change.

* **preprocess**: These are scripts for preprocessing fMRI time series and performing registration to the MNI template.

* **fcd_tools**: These are scripts for computing FCD feature maps.

* **polyML**: These are the scripts used for cross-validated classifications.

The scripts are provided "as is" for the purposes of reproducibility and transparency.

The following softwares and libraries are required for running the scripts:
Python 2.7, Pandas, Numpy, Matplotlib, Seaborn, Scickit-learn, Scikit-contrib Lightning, Nilearn, Nipy, Nipype, FSL, FreeSurfer, MATLAB


