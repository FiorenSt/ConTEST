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


###################
# PYTHON PACKAGES #
###################

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import os
import seaborn as sns
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

##############
# R PACKAGES #
##############

# import rpy2.robjects.packages as rpackages
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
# import rpy2.robjects




###########################################  REGRESSION  ##############################################
###########################################    MODELS    ##############################################


##########################
# CONTEST FOR REGRESSION #
##########################

def contest_reg(y_obs, x_obs, y_mod, y_obs_err, K=10000, seed=1, signif_lev=0.05, plot=False):
    # Observations size
    n = len(y_obs)

    # Estimate the test statistic
    t = numpy.sqrt(numpy.sum(((y_mod - y_obs) / y_obs_err) ** 2) / n)

    # Simulate K dataset with the model as ground truth
    sim = numpy.zeros((n, K))
    numpy.random.seed(seed)
    for i in range(n):
        sim[i, :] = y_mod[i] + numpy.random.normal(loc=0, scale=y_obs_err[i], size=K)

    # Estimate the test statistics for the simulated samples
    t_sim = numpy.sqrt(
        numpy.sum(((numpy.array([y_mod, ] * K).T - sim) / numpy.array([y_obs_err, ] * K).T) ** 2, axis=0) / n)

    # Calculate the P_value
    ecdf = ECDF(t_sim)
    p_value = 2 * numpy.min((ecdf(t),1-ecdf(t)))

    if (t > numpy.quantile(t_sim, q=1 - (signif_lev / 2))) | (t < numpy.quantile(t_sim, q=signif_lev / 2)):
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(x_obs, y_mod)
        plt.errorbar(x_obs, y_obs, yerr=y_obs_err, fmt='.k')
        plt.scatter(x_obs, y_obs, s=2, color='black')

        plt.subplot(1, 2, 2)
        s = sns.kdeplot(t_sim, fill=True, color="gray")
        plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
        plt.vlines(numpy.quantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')
        plt.vlines(numpy.quantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



###################################
# SMOOTHED CONTEST FOR REGRESSION #
###################################


############################
# ALLOW R TO ACCESS PYTHON #
############################

# In case of problems with R, run the tree lines below:
# os.environ['R_HOME'] = 'C:/Program Files/R/R-4.0.2'  -> Your installed R folder
# os.environ['R_USER'] = './miniconda3/envs/project5-consistencytest/lib/site-packages/'  -> Your python environment
# os.environ['R_LIBS_USER'] = "C:/Program Files/R/R-4.0.2/library/"  -> Your R packages library


######################
# INSTALL R PACKAGES #
######################


# def install_R_functions(packnames=('np')):
#     # import R's utility package
#     utils = rpackages.importr('utils')
#     # select a mirror for R packages
#     utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
#     # R package install
#     utils.install_packages(packnames)

# install_R_functions()



def smoothed_contest_reg(y_obs, x_obs, y_mod, y_obs_err, K=1000, seed=1, signif_lev=0.05, plot=False):
    # Observations size
    n = len(y_obs)

    np = rpackages.importr('np')

    x_obs = rpy2.robjects.r['matrix'](numpy.array(x_obs),ncol=x_obs.ndim)
    y_obs = rpy2.robjects.FloatVector(y_obs)

    rpy2.robjects.r('''
            # create a function `f`
            f <- function(y_obs, x_obs) {
                bw.fixed <- npregbw(ydat=y_obs,
                              xdat=x_obs,
                              regtype = "ll",
                              bwmethod = "cv.aic",
                              gradients = FALSE,
                              bwtype= 'fixed',
                              nmulti=1)
          
            model.fixed = npreg(bw.fixed)
          
            est_fixed=model.fixed$mean
            unc_fixed=model.fixed$merr
            
            return(cbind(est_fixed,unc_fixed))
            }
            ''')

    r_f = rpy2.robjects.globalenv['f']

    y_obs_smoothed = r_f(y_obs,x_obs)

    # Estimate the test statistic
    t = numpy.sqrt(numpy.sum(((y_mod - y_obs_smoothed[:,0]) / y_obs_smoothed[:,1]) ** 2) / n)

    # Simulate K dataset with the model as ground truth
    sim = numpy.zeros((n, K))
    numpy.random.seed(seed)

    for i in range(n):
        sim[i, :] = y_mod[i] + numpy.random.normal(loc=0, scale=y_obs_err[i], size=K)

    y_obs_smoothed_sim = numpy.zeros((n,K,2))
    for k in range(K):
        y_sim = rpy2.robjects.FloatVector(sim[:,k])
        y_obs_smoothed_sim[:,k,:] = r_f(y_sim, x_obs)


    # Estimate the test statistics for the simulated samples
    t_sim = numpy.sqrt(
        numpy.sum(((numpy.array([y_mod, ] * K).T - y_obs_smoothed_sim[:,:,0]) /y_obs_smoothed_sim[:,:,1]) ** 2, axis=0) / n)

    # Calculate the P_value
    ecdf = ECDF(t_sim)
    p_value = 2 * numpy.min((ecdf(t),1-ecdf(t)))

    if (t > numpy.quantile(t_sim, q=1 - (signif_lev / 2))) | (t < numpy.quantile(t_sim, q=signif_lev / 2)):
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(x_obs, y_mod)
        plt.plot(x_obs, y_obs_smoothed[:,0])
        plt.errorbar(x_obs, y_obs, yerr=y_obs_err, fmt='.k')
        plt.scatter(x_obs, y_obs, s=2, color='black')

        plt.subplot(1, 2, 2)
        s = sns.kdeplot(t_sim, fill=True, color="gray")
        plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
        plt.vlines(numpy.quantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')
        plt.vlines(numpy.quantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})




###########################################  DENSITY  ##################################################
###########################################  MODELS   ##################################################


########################
# CONTEST FOR OUTLIERS #
########################

def contest_outliers(mod, obs, K=1000, signif_lev=0.05, plot=False):
    n = len(obs)

    if mod.ndim == 1:
        mod = numpy.array(mod).reshape(-1, 1)
        obs = numpy.array(obs).reshape(-1, 1)

    g = KernelDensity(kernel='gaussian').fit(mod)
    t = -g.score(obs) / n

    # - simulate K dataset
    t_sim = numpy.zeros((K))

    for k in range(K):
        sim = g.sample(n)
        t_sim[k] = -g.score(sim) / n

    # Calculate the P_value
    ecdf = ECDF(t_sim)
    p_value = 2 * numpy.min((ecdf(t), 1 - ecdf(t)))

    if (t > numpy.quantile(t_sim, q=1 - (signif_lev / 2))) | (t < numpy.quantile(t_sim, q=signif_lev / 2)):
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))

    if plot:
        if obs.shape[1] == 1:
            plt.figure()
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=numpy.array(mod)[:, 0])
            plt.plot(numpy.array(obs)[:, 0], -0.00005 * numpy.ones(100), "|", color='black')

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
            plt.vlines(numpy.quantile(t_sim, q=1 - signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')
            plt.vlines(numpy.quantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

        if obs.shape[1] == 2:
            plt.figure()
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=numpy.array(mod)[:, 0], y=numpy.array(mod)[:, 1])
            plt.scatter(x=numpy.array(obs)[:, 0], y=numpy.array(obs)[:, 1], s=2, color='black')

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
            plt.vlines(numpy.quantile(t_sim, q=1 - signif_lev), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



##########################
# CONTEST FOR DENSITIES #
##########################

def contest_dens(mod, obs, K=1000, signif_lev=0.05, plot=False):
    n = len(obs)

    if mod.ndim == 1:
        mod = numpy.array(mod).reshape(-1, 1)
        obs = numpy.array(obs).reshape(-1, 1)

    g = KernelDensity(kernel='gaussian').fit(mod)

    f = KernelDensity(kernel='gaussian').fit(obs)

    # MCMC for the integral
    J = 1000

    ##simulate from g
    x_g = g.sample(J)

    d_g = numpy.exp(g.score_samples(x_g))
    d_f = numpy.exp(f.score_samples(x_g))

    t = sum(abs(d_f / d_g - 1)) / J

    # - simulate K dataset
    t_sim = numpy.zeros((K))

    for k in range(K):
        sim = g.sample(n)
        f_sim = KernelDensity(kernel='gaussian').fit(sim)
        d_f_sim = numpy.exp(f_sim.score_samples(x_g))
        t_sim[k] = sum(abs(d_f_sim / d_g - 1)) / J

    # P-value
    ecdf = ECDF(t_sim)
    p_value = 2 * numpy.min((ecdf(t), 1 - ecdf(t)))

    if t > numpy.quantile(t_sim, q=1 - signif_lev):
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4),  numpy.round(p_value,4)))

    if plot:

        if obs.shape[1] == 1:
            plt.figure()
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=numpy.array(mod)[:, 0])
            sns.kdeplot(x=numpy.array(obs)[:, 0])

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
            plt.vlines(numpy.quantile(t_sim, q=1 - signif_lev), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

        if obs.shape[1] == 2:
            plt.figure()
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=numpy.array(mod)[:, 0], y=numpy.array(mod)[:, 1])
            sns.kdeplot(x=numpy.array(obs)[:, 0], y=numpy.array(obs)[:, 1])

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors='orange')
            plt.vlines(numpy.quantile(t_sim, q=1 - signif_lev), ymin=0, ymax=s.dataLim.bounds[3], colors='darkblue')

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



if __name__ == '__main__':
    contest_reg()