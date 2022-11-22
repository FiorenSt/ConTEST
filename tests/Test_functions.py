from CONTEST import *
import numpy as np
from scipy import stats


###REGRESSION CHECK

n = 35
x = np.random.rand(n)

beta1 = -0.3
beta2 = 8
m = 2

model = np.exp(beta1 * x) * np.sin(beta2 * x) + m
err_model = model * .05

obs = np.zeros(n)

for i in range(n):
  obs[i] = model[i] + stats.multivariate_normal.rvs(mean=0, cov=(err_model[i]) ** 2, size=1)

err_obs = err_model

Test1 = contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,seed=4,plot=True)
Test2 = smoothed_contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,seed=4,plot=True)


#############DENSITY CHECK

## 1D

n=100
obs = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=n)
model = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=1000)

Test3 = contest_outliers(mod=model, obs=obs, K=10000, plot=True)
Test4 = contest_dens(mod=model, obs=obs, K=10000, plot=True)


## 2D

obs_2d =  stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=n)
model_2d = stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=1000)

Test3 = contest_outliers(mod=model_2d, obs=obs_2d, K=10000, plot=True)
Test4 = contest_dens(mod=model_2d, obs=obs_2d, K=10000, plot=True)
