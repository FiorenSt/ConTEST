from CONTEST import *
import random
import numpy as np
from scipy import stats

####FROM PIP INSTALL
#from ConTEST_V2.CONTEST import *

###REGRESSION CHECK

x = np.linspace(0,1,10000)

beta1 = -0.3
beta2 = 8
m = 2

real_model = np.exp(beta1*x)*np.sin(beta2*x) + m
err_real_model = real_model * .05 ##ACUTAL ERROR


#####OBSERVATIONS

N=100
sample = random.sample(range(10000),N)

###OBSERVATION FROM REAL MODEL
OBS = np.zeros(N)

for i in range(N):
  OBS[i] = real_model[sample[i]] + stats.multivariate_normal.rvs(mean=0, cov= (err_real_model[sample[i]])**2,size=1)

ERR_OBS = err_real_model[sample]


###MODEL IN EXAM
X = x[sample]  ###X AXIS
MODEL = real_model[sample] # - (REAL_MODEL/20)





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


Test1 = contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)
Test2 = smoothed_contest_reg(y_obs = obs, x_obs = x, y_mod = model, y_obs_err = err_obs, K=1000,plot=True)




#############DENSITY CHECK


## 1D

n=100

obs = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=n)
model = stats.multivariate_normal.rvs(mean=5, cov= [1.5],size=1000)

Test3 = contest_outliers(mod=model, obs=obs, K=1000, plot=True)
Test4 = contest_dens(mod=model, obs=obs, K=1000, plot=True)




## 2D
n=100
OBS = stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=n)
OBS

###MODEL IN EXAM (OBTAINED AS KDE FROM SIMULATED POINTS)
MODEL = stats.multivariate_normal.rvs(mean=[5,5], cov= [[1.5,.8],[.8,2.5]],size=1000)

#MODEL = KernelDensity(kernel='gaussian').fit(data)


Test3 = contest_outliers(mod=MODEL, obs=OBS, K=1000, plot=True)
Test4 = contest_dens(mod=MODEL, obs=OBS, K=1000, plot=True)
