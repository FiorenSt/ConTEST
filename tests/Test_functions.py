import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from CONTEST import *


####FROM PIP INSTALL
#from ConTEST.CONTEST import *


df = pd.read_csv("pythonDataCheck.csv", index_col=0)
df=df.sort_values('x_obs')

plt.figure(figsize=(8, 6))
plt.plot(df['x_obs'],df['y_mod'], linewidth=3)
plt.errorbar(df['x_obs'], df['y_obs'], yerr=df['uncertainties'], fmt='o', color='black',
             ecolor='black', elinewidth=1, capsize=3)
plt.show()

# Invoking the R function and getting the result
Test,Boot,Pvalue = ConTEST_reg(df,K=10)




#############DENSITY CHECK

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from CONTEST import *


df = pd.read_csv("pythonDataCheck_mod.csv", index_col=0)
df2 = pd.read_csv("pythonDataCheck_obs.csv", index_col=0)

# Invoking the R function and getting the result
Test,Boot,Pvalue = ConTEST_dens(df,df2,K=10)