<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMzAsInciOjEwMDAsImZzIjoxMzAsImZnYyI6IiNGRDhDMDMiLCJiZ2MiOiIjMDYwNTA1IiwidCI6MX0/Q29uVEVTVA/kg-second-chances-sketch.png width=50% height=50%>


<!--
[![DOI](https://zenodo.org/badge/440851447.svg)](https://zenodo.org/badge/latestdoi/440851447) 
<a href="https://ascl.net/2203.014"><img src="https://img.shields.io/badge/ascl-2203.014-blue.svg?colorB=262255" alt="ascl:2203.014" /></a>

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 

-->


# Description

ConTEST is a statistical test for assessing the consistency between observations and astrophysical models.
It uses a combination of non-parametric methods and distance measures to obtain a test
statistic that evaluates the closeness of the astrophysical model to the observations; hypothesis testing is then performed using bootstrap.

*ConTEST is written in R and wrapped in a python interface.* 

<img src=https://github.com/FiorenSt/ConTEST/blob/main/img/logo_contest_bkg.png width=15% height=15%>


## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)


# Installation


_Follow the instructions below to download and start using ConTEST._

1. Clone the repo
   ```sh
   pip intall ConTEST
   ```
   or
   ```sh
   git clone https://github.com/FiorenSt/ConTEST.git
   ```
 
 <br/>

  
2. Install the statistical software [R](https://www.r-project.org/).   
   R is only needed to run the core functions of ConTEST. No need to install R Studio.


<br/>

3. Test out ConTEST in Python!

<br/>

<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/MemeConTEST.png " width=80% height=80%>




# Dependencies:



### Python 3 (or superior)
* Numpy 1.20.3
* Pandas ...
* 


### R 4 (or superior)
* Np ...
* Mvtnorm
* Ks

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If a problem with the combiantion of packages occurs, raise an issue and we will help you solve it.



# Usage

ConTEST can be applied in different case scenarios depending on the nature of the model being tested. Below the two fundamental application on step-to-step examples.


## Regression models

0. Create synthetic Model, Observations and Uncertainties:

   ```sh
   x = 
   Mod =
   Obs = 
   Unc =

   ```

1. Use ConTEST_reg:
   
    ```sh
   import ConTEST

   Test = ConTEST_reg(x, Obs, Mod)
   
   ```


## Density models

0. Create synthetic Model and Observations:

   ```sh
   x = 
   y =
   
   Obs = 
   ```
 
1. Use ConTEST_dens:
   
    ```sh
   import ConTEST

   Test = ConTEST_dens(Obs, Mod)
   
   ```


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




# Credits

<!--
Credit goes to all the authors of the paper: 
**_AutoSourceID-Light. Fast Optical Source Localization via U-Net and Laplacian of Gaussian_**
-->
