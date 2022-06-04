########################################################################################################################
#                                                                                                                      #
#         ########      ## ## ##        ##         ##    ############   ## ########       #########     ############   #
#       ##            ##        ##      ## #       ##         ##        ##               ##                  ##        #
#      ##            ##          ##     ##  #      ##         ##        ##               ##                  ##        #
#     ##            ##            ##    ##   #     ##         ##        ##                ##                 ##        #
#     ##            ##            ##    ##    #    ##         ##        ## ########         ##               ##        #
#     ##            ##            ##    ##     #   ##         ##        ##                     ##            ##        #
#      ##            ##          ##     ##      #  ##         ##        ##                       ##          ##        #
#       ##            ##        ##      ##       # ##         ##        ##                       ##          ##        #
#         ########      ## ## ##        ##         ##         ##        ## ########     ##########           ##        #
#                                                                                                                      #
########################################################################################################################


import pandas as pd
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
import os
import rpy2.robjects.packages as rpackages


###CHANGE PATH FOR R

#os.environ['R_HOME'] = 'C:/Program Files/R/R-4.0.2'
#os.environ['R_USER'] = 'c:/users/fiore/miniconda3/envs/project5-consistencytest/lib/site-packages/' #path depends on where you installed Python. Mine is the site packages of the regular python installation, it could have been Anaconda
#os.environ['R_LIBS_USER'] = "C:/Users/fiore/OneDrive/Documenti/R/win-library/4.0/"
os.environ['R_LIBS_USER'] = "C:/Program Files/R/R-4.0.2/library/"

#np = rpackages.importr('np')


def install_R_functions(packnames = ('np')):
    # import R's utility package
    utils = rpackages.importr('utils')
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    # R package install
    utils.install_packages(packnames)



def load_data(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_r = robjects.conversion.py2rpy(df)
    return(df_r)



def ConTEST_reg(df,K=10,seed=10,signif_lev=0.05):

    np = rpackages.importr('np')

    ##load data
    data=load_data(df)

    # Defining the R script and loading the instance in Python
    r = robjects.r
    r['source']('ConTEST/src/ConTEST_V1/R_functions/ConTEST_reg.R')

    # Loading the function we have defined in R.
    ConTEST_reg = robjects.globalenv['ConTEST_reg']

    #Invoking the R function and getting the result
    df_result_r = ConTEST_reg(data[0],data[1],data[2],data[3],K=K,seed=seed,signif_lev=signif_lev)

    numpy2ri.activate()
    Test=df_result_r.rx2('Test_stat')
    Boot_star=df_result_r.rx2('Boot_stat')
    P_value=df_result_r.rx2('P_value')
    numpy2ri.deactivate()

    if (signif_lev >= P_value): print('Test statistic: {}, P-value: {} \nThe Null hypothesis in Rejected: the model is not consistent with the observations.'.format(Test,P_value))
    else: print('Test statistic: {}, P-value: {} \nThe Null hypothesis in Not Rejected: the model is consistent with the observations.'.format(Test,P_value))

    return(Test,Boot_star,P_value)




if __name__ == "__main__":
    from ConTEST.src.ConTEST_V1 import CONTEST
    df = pd.read_csv("pythonDataCheck.csv", index_col=0)

    # Invoking the R function and getting the result
    Test,Boot,Pvalue = ConTEST_reg(df,K=10)
