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
   git clone https://github.com/FiorenSt/ConTEST.git
   ```
   or
   ```sh
   pip intall ConTEST
   ```
 <br/>

  
2. (Optional) Install the statistical software [R](https://www.r-project.org/).   
   R is only needed to run Smoothed ConTEST. (No need to install R Studio)

<br/>

3. Use ConTEST in Python!

<br/>

<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/MemeConTEST.png " width=80% height=80%>



# Dependencies:


### Python 3 (or superior)
* Numpy 1.21.6
* Pandas 1.4.2
* Scipy 1.7.1
* Matplotlib 3.3.4
* Seaborn 0.11.2
* Rpy2 3.5.2 (For R and Python interaction)

### R 3.6.0 (or superior)
* Np 0.60
* Ks 1.13.5

This combination of package versions works on most Linux and Windows computers, however other package versions may also
work.
If a problem with the combination of packages occurs, raise an issue, and we will help you solve it.


# Usage

ConTEST can be applied in different case scenarios depending on the nature of the model being tested. 
<br/>
For more details check out the paper: _Stoppa et al., in preparation_

There are 4 fundamental functions in ConTEST:

- ConTEST for regression: Test the consistency of a model with respect to an observed dataset and their uncertainties
- Smoothed ConTEST for regression: Test the consistency of a model with respect to an observed dataset and 
  their uncertainties
- ConTEST for outliers: Test if an observed sample is likely to come from a density model (or a simulated dataset)
- ConTEST for densities: Test the consistency of a density model (or a simulated dataset) with respect to an observed 
  dataset 



## Regression models

0. Create synthetic model, observations, and uncertainties to test the functions:

   ```sh
   # random sample 
   n=100
   x = np.random.rand(n)
   
   # synthetic model 
   beta1 = -0.3
   beta2 = 8
   m = 2
   model = np.exp(beta1*x)*np.sin(beta2*x) + m
   
   # error function (Not known in real scenarios)
   err_model = model * .05
   
   # sample observations from the model with the correct uncertainties
   obs = np.zeros(N)
      for i in range(N):
     obs[i] = model[i] + stats.multivariate_normal.rvs(mean=0, cov=(err_model[i])**2,size=1)
   
   # assign correct uncertainties to the observations
   err_obs = err_model
   ```

### ConTEST for regression
 
   ```sh
   Test1 = contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)
   ```
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/ConTESTforRegression.png " width=80% height=80%>


### Smoothed ConTEST for regression
 
   ```sh
   Test2 = smoothed_contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)   
   ```
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/SmoothedConTESTforRegression.png " width=80% height=80%>


## Density models

0. Create synthetic model and observations:

   ```sh
   n=100
   
   #1D example
   obs = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=n)
   model = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=1000)
    
   #2D example
   obs_2d =  stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=n)
   model_2d = stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=1000)

### ConTEST for outliers 

   ```sh
   Test3 = contest_outliers(mod=model, obs=obs, K=10000, plot=True)
   Test3 = contest_outliers(mod=model_2d, obs=obs_2d, K=10000, plot=True)
   ```
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/ConTESTforOutliers1D.png " width=80% height=80%>
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/ConTESTforOutliers2D.png " width=80% height=80%>



### ConTEST for densities 

   ```sh
   Test4 = contest_dens(mod=model, obs=obs, K=10000, plot=True)   
   Test4 = contest_dens(mod=model_2d, obs=obs_2d, K=10000, plot=True)

   ```
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/ConTESTforDensities1D.png " width=80% height=80%>
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/ConTESTforDensities2D.png " width=80% height=80%>




# Credits
