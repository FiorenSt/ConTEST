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
   
   n=100
   x = np.random.rand(n)
   
   beta1 = -0.3
   beta2 = 8
   m = 2
   
   model = np.exp(beta1*x)*np.sin(beta2*x) + m
   err_model = model * .05
   
   ###OBSERVATION FROM REAL MODEL
   obs = np.zeros(N)
   
   for i in range(N):
     obs[i] = model[i] + stats.multivariate_normal.rvs(mean=0, cov=(err_model[i])**2,size=1)
   
   err_obs = err_model

   ```

1. Use ConTEST_reg:
   
    ```sh
   Test1 = contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)
   Test2 = smoothed_contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)   
   ```


## Density models

0. Create synthetic Model and Observations:

   ```sh
   n=100
   obs = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=n)
   model = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=1000)
 
1. Use ConTEST_dens:
   
    ```sh
   import ConTEST
   Test3 = contest_outliers(mod=model, obs=obs, K=1000, plot=True)
   Test4 = contest_dens(mod=model, obs=obs, K=1000, plot=True)   
   ```




# Credits
