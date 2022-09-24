from CONTEST import *


####FROM PIP INSTALL
#from ConTEST_V2.CONTEST import *

###REGRESSION CHECK
data = numpy.loadtxt(
    'C:/Users/fiore/Desktop/UNI/Projects/Project5-ConsistencyTest/SCRIPTS/REGRESSION/spectrum_GOOD_fit.dat', skiprows=1)

y_obs = data[:, 1]
x_obs = data[:, 0]
y_mod = data[:, 2]
y_obs_err = data[:, 3]

Test1 = contest_reg(y_obs, x_obs, y_mod, y_obs_err,K=1000,plot=True)
Test2 = smoothed_contest_reg(y_obs, x_obs, y_mod, y_obs_err,K=1000,plot=True)


#############DENSITY CHECK

df = pd.read_csv("pythonDataCheck_mod.csv", index_col=0)
df2 = pd.read_csv("pythonDataCheck_obs.csv", index_col=0)

Test3 = contest_outliers(mod=df['mod.2'], obs=df2['obs.2'], K=1000, plot=True)
Test4 = contest_dens(mod=df['mod.2'], obs=df2['obs.2'],K=1000, plot=True)


