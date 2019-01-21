# REPLACEMENT FOR NIPYPEŚ IO INTERFACE

We modified nipypes io interface to accept modernly formated string as inputs.
For we did:
replace: filledtemplate = template % tuple(argtuple)
with: filledtemplate = template.format(*argtuple) 

in all instances.

The io.py file found here should be put in nipype/interfaces/io.py

