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


import argparse
from Load_data import *

#parser = argparse.ArgumentParser()
#parser.add_argument("-y_mod", type=float)
#parser.add_argument("-x_obs", type=float)
#parser.add_argument("-y_obs", type=float)
#parser.add_argument("-uncertainties", type=float)
#parser.add_argument("-K", type=int)
#parser.add_argument("-seed", type=int)
#parser.add_argument("-signif_lev", type=float)
#parser.add_argument("--mode", default='client')
#parser.add_argument("--port", default=52162)
#args = parser.parse_args()


import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

def ConTEST_reg(df,K=10,seed=10,signif_lev=0.05):

    ##load data
    data=load_data(df)

    # Defining the R script and loading the instance in Python
    r = robjects.r
    r['source']('ConTEST/R_functions/ConTEST_reg.R')

    # Loading the function we have defined in R.
    ConTEST_reg = robjects.globalenv['ConTEST_reg']

    #Invoking the R function and getting the result
    df_result_r = ConTEST_reg(data[0],data[1],data[2],data[3],K=K,seed=seed,signif_lev=signif_lev)

    numpy2ri.activate()
    Test=df_result_r.rx2('Test_stat')
    Boot_star=df_result_r.rx2('Boot_stat')
    P_value=df_result_r.rx2('P_value')
    numpy2ri.deactivate()

    return(Test,Boot_star,P_value)




if __name__ == "__main__":
#    import pandas as pd
#    from rpy2.robjects.conversion import localconverter
#    from rpy2.robjects import pandas2ri

    df = pd.read_csv("pythonDataCheck.csv", index_col=0)

    # Invoking the R function and getting the result
    Test,Boot,Pvalue = ConTEST_reg(df,K=1000)
    print(Test, Pvalue)