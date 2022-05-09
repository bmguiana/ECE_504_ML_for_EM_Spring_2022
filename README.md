# Machine Learning for Detection of Modal Configurations in Rectangular Waveguides
### Author: Brian Guiana
### Date: 9 May 2022

This code was written for a class project in the course entitled ECE 504: "Machine Learning for Electromagnetics" that was taught in Spring 2022 by Prof. Zadehgol at the University of Idaho in Moscow, Idaho. [4]

## Overview
This project and the code herein applies some electromagnetics (micro-wave, mm-wave) fundamentals to machine learning classification.
- Idea data are generated using equations and methods [2], [3]
- Noise is added to ideal data using [7] with methods from [6]
- Noisy data are used to train and test a binary and nonbinary classifier machine learning model with [1] [5]
- Data are saved as a pickled dictionary (.npy format). Sizes vary depending on resolution and number of samples, but typically require memory and disk space on the scale of 5-20 GB.
- Currently available noise figures are: Uniform, Gaussian (Normal), Uncorrelated Exponential, and Correlated Exponential. These add only milliseconds (compared to control) to the per sample generation time, with the exception of Correlated Exponential noise which add seconds. Expect a few seconds to generate 10,000 samples with the former and around 10 minutes for the latter.
- Arbitrary signal to noise ratio values may be assigned. Individual files are saved for each.
- Arbitrary waveguide parameters can be assigned. Loss between points is handled through the initial amplitude setting.
- One Classifier is provided, but the function using it can have the model swapped out for any other classification model within Scikit-learn.
- This code is not meant to be general use but as a proof of concept for the methods of applying classification to waveguide mode identification.

## Licensing
This repository and all code herein is licensed under the GNU GPL v3.0 license.

## Files
This repository contains 3 files:
- gen_data_final.py: Generates data for use with train_and_test.py
- train_and_test.py: Trains and tests a binary and nonbinary classification model
- reamde.md: This file

## Code
- gen_data_final.py: uses user provided inputs to generate data files.
- train_and_test.py: uses data files from gen_data_final.py to train and test machine learning classifier models

## Run instructions
1. Set any custom input parameters in gen_data_final.py
1. Run gen_data_final.py
2. Set the input parameters in train_and_test.py from gen_data_final.py
3. Run train_and_test.py

## Input parameters
Parameters are given in the form:
VARIABLE_NAME: Description (units) (data type)
- SNR: Signal to noise ratio (unitless from V/m//V/m) (float)
- NF: Noise figure type, 0 to 4 inclusive, where 0: No noise, 1: Uniform noise, 2: exponential noise, 3: gaussian noise, 4: correlated noise (None) (int)
- num_samples: Total number of samples in the data set (samples) (int)
- pixels: Image size (pixels x pixels) (int)
- a: Waveguide width (m) (float)
- b: Waveguide height (m) (float)
- f: Source frequency (Hz) (float)
- eps_rel: Relative permittivity (unitless from F/m//F/m) (float)
- A: TE mode base amplitude (V/m) (float)
- B: TM mode base amplitude (A/m) (float)

## Output parameters
A binary classifier and nonbinary classifier machine learning model.
- noise_type_snr.npy: The data of Ex and Ey field components and their corresponding labels, used for classifier training/testing.

## Usage
This project was designed to generate rectangular waveguide data and use it to train a classifying machine learning model.

## Python version info
- Python: 3.9.12
- iPython: 8.2.0
- Numpy: 1.20.3
- Pyspeckle: 0.4.1
- Scikit-Learn: 1.0.2

## References
```
[1] Aurelien Geron, Hands-On Machine Learning with Scikit-Learn,
  Keras, & TensorFlow. O'Reilly Media, Inc., Second edition, Sept. 2019.

[2] D.M. Pozar, Microwave Electronics. Wiley, USA, 4th edition, 2012.

[3] C.A. Balanis, Advanced Engineering Electromagnetics.
  Wiley, USA, 1st edition, 1989.

[4] ECE 504: Machine Learning for Electromagnetics.
  University Course, Spring 2022, University of Idaho, Moscow, ID.

[5] Aurelien Geron, et al, handson-ml2 [Online] github.com/ageron/handson-ml2/.
  10 Feb. 2022.

[6] M.~Deserno, How to generate exponentially correlated Gaussian random numbers.
  August 2002.

[7] Pyspeckle, https://pyspeckle2.readthedocs.io/en/latest/#}.
  Accessed: Oct. 15, 2021.
```
