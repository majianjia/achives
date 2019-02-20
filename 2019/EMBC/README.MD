# About
This folder is the paper archive for the source code of EMBC 2019.

The paper (under reviewing) is about implementing a complex neurual network classifier into microcontroller for activitie recognition. 

`/models` shows the Inception classifier which is trained with keras and tensorflow

`/mcu` includes the c files for rebuilding classifier on microcontroller using Neural Netowork on Microcontroller (NNoM) framework

NNoM is a fast developing lib specifically for running and evaluating quantised NN model directly on microcontroller. 

This achieve does not include NNoM.
However, the latest version can be found in [NNoM's repository](https://github.com/majianjia/nnom)
