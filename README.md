# ConTEST

<!--
<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/kg-second-chances-sketch.png>


[![DOI](https://zenodo.org/badge/440851447.svg)](https://zenodo.org/badge/latestdoi/440851447) 
<a href="https://ascl.net/2203.014"><img src="https://img.shields.io/badge/ascl-2203.014-blue.svg?colorB=262255" alt="ascl:2203.014" /></a>

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 

-->


# Description
Nonparametric consistency test between observations and astrophysical models


## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Credits](#credits)


# Installation

<!--
_Follow the instructions below to download and start using ASID-L._

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/ConTEST.git
   ```
2. Download the Zenodo folder for training/test/validation sets    
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5902893.svg)](https://doi.org/10.5281/zenodo.5902893)

3. Save the files in a folder "TrainingSet" and include the folder in the ASID-L repository
4. Create an empty folder "RESULTS" 
-->

# Dependencies:

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/img/MemeConTEST.png " width=50% height=50%>

ConTEST is written in R and wrapped up in a python interface. Below the required packages:

* Python 3 (or superior)
* R 4 (or superior)
* Numpy 1.20.3
* Pandas ...
* 

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If a problem with the combiantion of packages occurs, raise an issue and we will help you solve it.



# Usage
<!--

The use of the pre-trained ASID-L is straight forward: 

```
python ASID-L.py
```

It loads a .fits image and the pre-trained model, and it outputs a catalog 'coordinates.txt' in the folder 'RESULTS'.

**Other parameters:**
 
-DATA_PATH './TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits'  **_(path of the .fits image)_**

-MODEL_PATH './MODELS/TrainedModel.h5'   **_(path of the model)_**

-demo_plot   **_(shows a plot with an optical patch superimposed with the locations of the sources in red)_**

-CPUs  **_(number of CPUs for parallel processing)_**

Here an example,
```
python ASID-L.py -DATA_PATH './TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits' -MODEL_PATH './MODELS/TrainedModel.h5' -demo_plot
```

### Train U-Net from scratch

 To train the U-Net without additional changes run:
 ```
 python ASID-L.py -train_model
 ```
 You will find the trained model in the folder '/MODELS/FROM_SCRATCH'. You can then run the pre-trained version of ASID-L with -MODEL_PATH your new trained model.
 
**Other parameters:**

-snr_threshold **_(SNR cut-off for the training set)_** 

-epochs **_(the number of epochs)_**


-->


# Features

<!--

An open question that we want to address in the future is how the resolution of the images affects the localization results.
A first promising test can be found below, where we applied ASID-L, trained on MeerLICHT images, to images from the Hubble Space Telescope. The latter has a Full-Width at Half-
Maximum (FWHM) PSF of about 0.11 arcseconds, much better than the 2-3 arcseconds of MeerLICHT.

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTField10396.png " >

Star cluster image retrieved from the Hubble Space Telescope archive (GO-10396, PI: J.S. Gallagher). The red circles in the zoomed windows are the locations of the sources identified by ASID-L.

Although this is an early study, it appears that ASID-L is capable of localizing  sources without the need to re-train the U-Net on HST images. The main difference between MeerLICHT and HST, the resolution of the images, does not seem to affect the results of the method. 
-->

# Credits
Credit goes to all the authors of the paper: 

**_AutoSourceID-Light. Fast Optical Source Localization via U-Net and Laplacian of Gaussian_**