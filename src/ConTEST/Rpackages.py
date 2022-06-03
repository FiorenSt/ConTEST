import pandas as pd
import os
from rpy2.robjects.conversion import localconverter


import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
import rpy2.robjects.packages as rpackages


# see the following 2 lines
#os.environ['R_HOME'] = 'C:/Program Files/R/R-4.0.2'
#os.environ['R_USER'] = 'c:/users/fiore/miniconda3/envs/project5-consistencytest/lib/site-packages/' #path depends on where you installed Python. Mine is the site packages of the regular python installation, it could have been Anaconda
#os.environ['R_LIBS_USER'] = "C:/Users/fiore/OneDrive/Documenti/R/win-library/4.0/"
os.environ['R_LIBS_USER'] = "C:/Program Files/R/R-4.0.2/library/"
#import rpy2.situation
#for row in rpy2.situation.iter_info():
#    print(row)

np = rpackages.importr('np')


def install_R_functions(packnames = ('np')):

    # import R's utility package
    utils = rpackages.importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list

    # R package install
    utils.install_packages(packnames)



if __name__ == "__main__":
    install_R_functions()