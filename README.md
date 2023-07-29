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
 

<img src=https://github.com/FiorenSt/ConTEST/blob/main/img/logo_contest_bkg.png width=15% height=15%>


## Table of Contents 
- [Step-by-step setup](#step-by-step-setup)
- [Tutorial](#tutorial)


# Step-by-step setup

_Follow the instructions below to install and start using ConTEST in Python._

1. Install ConTEST:
   ```sh
   pip install ConsistencyTEST
   ```
   or git clone the repository:
    ```sh
   git clone https://github.com/FiorenSt/ConTEST.git
   ```
   

 <br/>

2. (Optional) Install the statistical software [R](https://www.r-project.org/). R is required only if you plan to use the `smoothed_contest_reg()` function. This function employs local linear regression using the `np` package in R.
<br/>

3. (Optional) To ensure that Python can access R's libraries, run the three lines below in Python (of course, modify to match your folders). This step is only necessary if you intend to use the `smoothed_contest_reg()` function:

   ```sh
    import os
    os.environ['R_HOME'] = '~/Program Files/R/R-4.0.2'  #-> Your installed R folder
    os.environ['R_USER'] = '~/Miniconda3/envs/ConsistencyTest/lib/site-packages/'  #-> Your python environment
    os.environ['R_LIBS_USER'] = "~/Program Files/R/R-4.0.2/library/"  #-> Your R packages library
   ```
<br/>


4. Install Python dependencies. Note that the `rpy2` package, which facilitates interaction between R and Python, is required only if you plan to use the `smoothed_contest_reg()` function:
   ```sh
   pip intall matplotlib
   pip intall numpy
   pip intall pandas
   pip intall scipy
   pip intall seaborn
   pip intall rpy2
   ```
 <br/>


5. (Optional) If this is the first time you use ConTEST and you plan to use the `smoothed_contest_reg()` function, you need to install the R package used in Smoothed ConTEST. In Python, simply run:

   ```sh
    def install_R_functions(packnames=('np')):
        # import R's utility package
        utils = rpackages.importr('utils')
        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        # R package install
        utils.install_packages(packnames)
    
    install_R_functions()
   ```

3. Use ConTEST in Python! Follow the tutorial below for more information about the individual functions.

<br/>


<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/MemeConTEST.png " width=80% height=80%>

<br/>




# Dependencies:

The following combination of package versions works on most Linux and Windows computers, however other package versions may also
work. If a problem with the combination of packages occurs, raise an issue, and we will help you solve it.


### Python 3 (or superior)
* Numpy 1.21.6
* Pandas 1.4.2
* Scipy 1.7.1
* Matplotlib 3.3.4
* Seaborn 0.11.2
* Rpy2 3.5.2 (For R and Python interaction)

### R 3.6.0 (or superior)
* Np 0.60



# Tutorial

ConTEST can be applied in different case scenarios depending on the nature of the model being tested. 
<br/>
For more details check out the paper: _Stoppa et al., in preparation_

There are 4 fundamental functions in ConTEST:

- ConTEST for regression: Test the consistency of a model with respect to an observed dataset and their uncertainties
- Smoothed ConTEST for regression (requires R and the `rpy2` Python package): Test the consistency of a model with respect to an observed dataset and 
  their uncertainties
- ConTEST for outliers: Test if an observed sample is likely to come from a density model (or a simulated dataset)
- ConTEST for densities: Test the consistency of a density model (or a simulated dataset) with respect to an observed 
  dataset 


### Intro script 

   ```sh
    # ensure that Python can access R
    import os
    os.environ['R_HOME'] = '~/Program Files/R/R-4.0.2'  #-> Your installed R folder
    os.environ['R_USER'] = '~/Miniconda3/envs/ConsistencyTest/lib/site-packages/'  #-> Your python environment
    os.environ['R_LIBS_USER'] = "~/Program Files/R/R-4.0.2/library/"  #-> Your R packages library
    
    # load contest functions
    from ConTEST.CONTEST import contest_reg, smoothed_contest_reg, contest_outliers, contest_dens
   ```


## Regression models

Create synthetic model, observations, and uncertainties to test the functions:

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


### Smoothed ConTEST for regression (requires R and the `rpy2` Python package)
 
   ```sh
   Test2 = smoothed_contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)   
   ```
<img src="https://github.com/FiorenSt/ConTEST/blob/main/img/SmoothedConTESTforRegression.png " width=80% height=80%>


## Density models

Create synthetic model and observations:

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




