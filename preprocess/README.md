### Directory contents:

**dcm_conversion**: Script for converting raw dicom files into nifti format.

**do_rsfmri.py**: Script for preprocessing resting-state fMRI time series

**registration_setup.py**: Script for registration of T1W anatomical to MNI space, fMRI to T1W and MNI space.


**conversion_tools.py**: Functions for creation of Nipype pipelines that are used in dcm_conversion.py

**resting_wf.py**: Functions for creation of Nipype pipelines that are used in do_rsfmri.py, modified from the example that came with Nipype
on nipype.workflows.rsfmri.fsl.resting.create_resting_preproc (also http://nipy.org/nipype/0.9.2/users/examples/rsfmri\_fsl\_compcorr.html)

**registration_struct_wf.py**: Functions for creation of Nipype pipelines that are used in registration_setup.py for registration of T1W to MNI

**registration_func_wf.py**: Functions for creation of Nipype pipelines that are used in registration_setup.py for registration of fMRI to T1W and MNI

