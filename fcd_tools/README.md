### Directory contents:

*This MATLAB code is the one previously used by Rish et. al. (2013) for computing voxel level funcional connectivity density maps (degree maps)*

*Rish, I., Cecchi, G., Thyreau, B., Thirion, B., Plaze, M., Paillere-martinot, M.-L., et al. (2013). Schizophrenia as a Network Disease: Disruption of Emergent Brain Function in Patients with Auditory Hallucinations. PloS One, 8(1), e50625. http://doi.org/10.1371/journal.pone.0050625*

#### Scripts (in the order in which they are typically called)
**export_subjects_list_for_MLtools.py**: Script for making a subject list to be used by the matlab code.

**make_functional_mask.m**: Script for making a brain mask to be used.

**detect_func_mask_outliers.m**: To filter out poorly registered subjects.

**main_TON_rsfmri.m**: Script to make the feature maps.

**post_process_feature_matrix.m**: Used for taking the log and/or smoothing.

*All other files are functions called by these main scripts and not run on their own*
