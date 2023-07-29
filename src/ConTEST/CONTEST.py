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
import seaborn as sns
from scipy import stats


############################
# ALLOW R TO ACCESS PYTHON #
############################

# import os
#os.environ['R_HOME'] = '~/Program Files/R/R-4.0.2'  #-> Your installed R folder
# os.environ['R_USER'] = '~/Miniconda3/envs/ConsistencyTest/lib/site-packages/'  #-> Your python environment
# os.environ['R_LIBS_USER'] = "~/Program Files/R/R-4.0.2/library/"  #-> Your R packages library


try:
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects.numpy2ri
    import rpy2.robjects
    rpy2_imported = True
except ImportError:
    rpy2_imported = False
    print("Warning: rpy2 module not found. The function smoothed_contest_reg() will not work without it. If you need this function, please install rpy2.")
    

###########################################  REGRESSION  ##############################################
###########################################    MODELS    ##############################################


##########################
# CONTEST FOR REGRESSION #
##########################

def contest_reg(y_obs, x_obs, y_mod, y_obs_err, K=10000, seed=1, signif_lev=0.05, plot=False):

    # Observations size
    n = len(y_obs)

    # Check if input is a vector or a matrix
    if y_obs_err.ndim == 1:
        y_obs_err = y_obs_err
        # Estimate the test statistic for 1D case
        t = numpy.sqrt(numpy.sum(((y_mod - y_obs) / y_obs_err) ** 2) / n)

        # Simulate K dataset with the model as ground truth and the observed uncertainties for 1D case
        sim = numpy.zeros((n, K))
        numpy.random.seed(seed)
        for i in range(n):
            sim[i, :] = y_mod[i] + stats.multivariate_normal.rvs(mean=0, cov=(y_obs_err[i]) ** 2, size=K)

        # Estimate the test statistics for the simulated samples for 1D case
        t_sim = numpy.sqrt(
            numpy.sum(((numpy.array([y_mod, ] * K).T - sim) / numpy.array([y_obs_err, ] * K).T) ** 2, axis=0) / n)

    else:
        y_obs_cov = y_obs_err
        # Estimate the test statistic for 2D case
        t = numpy.sqrt(numpy.dot((y_mod - y_obs).T, numpy.linalg.inv(y_obs_cov)).dot(y_mod - y_obs) / n)

        # Simulate K dataset with the model as ground truth and the observed uncertainties as covariance matrix for 2D case
        sim = numpy.zeros((n, K))
        numpy.random.seed(seed)
        for i in range(K):
            sim[:, i] = stats.multivariate_normal.rvs(mean=y_mod, cov=y_obs_cov)

        # Estimate the test statistics for the simulated samples for 2D case
        t_sim = numpy.sqrt(numpy.sum((y_mod[:, None] - sim).T * numpy.linalg.inv(y_obs_cov).dot(y_mod[:, None] - sim).T, axis=1) / n)

    # Calculate the P_value
    p_value = 2 * min(numpy.mean(t_sim <= t), numpy.mean(t_sim >= t))

    if p_value < signif_lev:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))

    if plot:
        plt.figure(figsize=(12,6))

        order = numpy.argsort(x_obs)
        plt.subplot(1, 2, 1)
        plt.plot(x_obs[order], y_mod[order],color="#4E84C4", label='Model')
        plt.errorbar(x_obs[order], y_obs[order], yerr=y_obs_err[order], fmt='.k')
        plt.scatter(x_obs[order], y_obs[order], s=2, color='black')
        plt.legend()

        plt.subplot(1, 2, 2)
        s = sns.kdeplot(t_sim, fill=True, color="gray")
        plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors="#D16103",linestyle='--', label='Test stat.')
        plt.vlines(numpy.nanquantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors="#293352",
                   linestyle='--', label='97.5%')
        plt.vlines(numpy.nanquantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3],
                   colors="#293352",linestyle='--', label='2.5%')
        plt.legend()

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



###################################
# SMOOTHED CONTEST FOR REGRESSION #
###################################

def smoothed_contest_reg(y_obs, x_obs, y_mod, y_obs_err, K=1000, seed=1, signif_lev=0.05, bwtype='fixed', plot=False, verbose=0):

    rpy2.robjects.numpy2ri.activate()

    # Observations size
    n = len(y_obs)

    # Load package for Local Linear Regression
    np = rpackages.importr('np')

    # Transform observations and model to format suitable for R
    x_obs_r = rpy2.robjects.r['matrix'](numpy.array(x_obs),ncol=x_obs.ndim)
    y_obs_r = rpy2.robjects.FloatVector(y_obs)

    # R script for the LLR
    rpy2.robjects.r('''
            # create a function `f`
            f <- function(y_obs_r, x_obs_r, bwtype) {
                bw.fixed <- npregbw(ydat=y_obs_r,
                              xdat=x_obs_r,
                              regtype = "ll",
                              bwmethod = "cv.aic",
                              gradients = FALSE,
                              bwtype= bwtype,
                              nmulti=1)
          
            model.fixed = npreg(bw.fixed)
          
            est_fixed=model.fixed$mean
            unc_fixed=model.fixed$merr
            
            return(cbind(est_fixed,unc_fixed))
            }
            ''')

    # Run the LLR on the original dataset
    r_f = rpy2.robjects.globalenv['f']
    y_obs_smoothed = r_f(y_obs_r, x_obs_r, bwtype)

    # Estimate the test statistic
    t = numpy.sqrt(numpy.sum(((y_mod - y_obs_smoothed[:,0]) / y_obs_smoothed[:,1]) ** 2) / n)

    # Simulate K dataset with the model as ground truth and the observed uncertainties as covariance matrix
    sim = numpy.zeros((n, K))
    numpy.random.seed(seed)
    for i in range(n):
        sim[i, :] = y_mod[i] + stats.multivariate_normal.rvs(mean=0, cov= (y_obs_err[i])**2,size=K)

    # Run the LLR on the simulated samples
    y_obs_smoothed_sim = numpy.zeros((n,K,2))
    for k in range(K):
        y_sim = rpy2.robjects.FloatVector(sim[:,k])
        y_obs_smoothed_sim[:,k,:] = r_f(y_sim, x_obs_r, bwtype)
        if verbose == 1: print('Iteration:' + k)

    # Estimate the test statistics for the simulated samples
    t_sim = numpy.sqrt(
        numpy.sum(((numpy.array([y_mod, ] * K).T - y_obs_smoothed_sim[:,:,0]) / y_obs_smoothed_sim[:,:,1]) ** 2, axis=0) / n)

    # Calculate the P_value
    p_value = 2 * min(numpy.mean(t_sim <= t), numpy.mean(t_sim >= t))

    if p_value < signif_lev:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))

    if plot:
        plt.figure(figsize=(12,6))
        order = numpy.argsort(x_obs)

        plt.subplot(1, 2, 1)
        plt.plot(x_obs[order], y_mod[order],color="#4E84C4", label='Model')
        plt.plot(x_obs[order], y_obs_smoothed[:,0][order],color="#D16103", label='Obs.')
        plt.plot(x_obs[order], y_obs_smoothed[:, 0][order]+y_obs_smoothed[:, 1][order],'--',color="#D16103")
        plt.plot(x_obs[order], y_obs_smoothed[:, 0][order]-y_obs_smoothed[:, 1][order],'--',color="#D16103")
        plt.errorbar(x_obs[order], y_obs[order], yerr=y_obs_err[order], fmt='.k')
        plt.scatter(x_obs[order], y_obs[order], s=2, color='black')
        plt.legend()

        plt.subplot(1, 2, 2)
        s = sns.kdeplot(t_sim, fill=True, color="gray")
        plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], color="#D16103",linestyle='--', label='Test stat.')
        plt.vlines(numpy.nanquantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], color="#293352",
                   linestyle='--', label='97.5%')
        plt.vlines(numpy.nanquantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3], color="#293352",
                   linestyle='--', label='2.5%')
        plt.legend()

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})




###########################################  DENSITY  ##################################################
###########################################  MODELS   ##################################################


########################
# CONTEST FOR OUTLIERS #
########################


def contest_outliers(mod, obs, K=1000, signif_lev=0.05, plot=False):

    # Sample size
    n = obs.shape[0]

    # Transformation for scipy multivariate Kernel Density Estimation
    if obs.ndim == 2:
        mod = mod.T
        obs = obs.T

    # Estimate the model's density
    g = stats.gaussian_kde(mod)

    # Estimate the test statistic
    t = -numpy.mean(g.logpdf(obs))

    # Simulate K dataset and estimate their test statistics
    t_sim = numpy.zeros((K))
    for k in range(K):
        sim = g.resample(n)
        t_sim[k] = -numpy.mean(g.logpdf(sim))

    # Calculate the P_value
    p_value = 2 * min(numpy.mean(t_sim <= t), numpy.mean(t_sim >= t))

    if p_value < signif_lev:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Rejected: the model is not consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))
    else:
        print(
            'Test statistic: {}, P-value: {} \nThe Null Hypothesis in Not Rejected: the model is consistent with the '
            'observations.'.format(
                numpy.round(t, 4), p_value))

    if plot:
        if obs.ndim == 1:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=mod, levels=[0.25,0.50,0.75], color="#4E84C4", linestyle='--', label='Model')
            plt.plot(obs, -0.00005 * numpy.ones(obs.shape[0]), "|", color='black')
            plt.legend()

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors="#D16103", linestyle='--', label='Test stat.')
            plt.vlines(numpy.nanquantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3],
                       colors="#293352", linestyle='--', label='97.5%')
            plt.vlines(numpy.nanquantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], colors="#293352",
                       linestyle='--', label='2.5%')
            plt.legend()


        if obs.ndim == 2:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=mod[0,:], y=mod[1,:], levels=[0.25,0.50,0.75], color="#4E84C4", linestyles='--',
                        label='Model')
            plt.scatter(x=obs[0,:], y=obs[1,:], s=2, color='black')
            plt.legend()

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], color="#D16103", linestyle='--', label='Test stat.')
            plt.vlines(numpy.nanquantile(t_sim, q=1 - (signif_lev / 2)), ymin=0, ymax=s.dataLim.bounds[3],
                       color="#293352", linestyle='--', label='97.5%')
            plt.vlines(numpy.nanquantile(t_sim, q=signif_lev / 2), ymin=0, ymax=s.dataLim.bounds[3], color="#293352",
                       linestyle='--', label='2.5%')
            plt.legend()

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



##########################
# CONTEST FOR DENSITIES #
##########################

def contest_dens(mod, obs, K=1000, signif_lev=0.05, plot=False):
    # Sample size
    n = obs.shape[0]

    # Transformation for scipy multivariate Kernel Density Estimation
    if obs.ndim == 2:
        mod = mod.T
        obs = obs.T

    # Estimate the model's density
    g = stats.gaussian_kde(mod)

    # Estimate the observations' density
    f = stats.gaussian_kde(obs)

    # MCMC to estimate the test statistic
    J = 1000
    x_g = g.resample(J)
    d_g = g.pdf(x_g)
    d_f = f.pdf(x_g)

    # Estimate the test statistics
    t = sum(abs(d_f / d_g - 1)) / J

    # Simulate K dataset and estimate their test statistics
    t_sim = numpy.zeros((K))
    for k in range(K):
        sim = g.resample(n)
        f_sim = stats.gaussian_kde(sim)
        d_f_sim = f_sim.pdf(x_g)
        t_sim[k] = sum(abs(d_f_sim / d_g - 1)) / J

    # P-value
    p_value = numpy.mean(t_sim >= t)

    if p_value < signif_lev:
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

        if obs.ndim == 1:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=mod, levels=[0.25,0.50,0.75], color="#4E84C4", linestyle='--', label='Model')
            sns.kdeplot(x=obs, levels=[0.25,0.50,0.75], color="#D16103", label='Obs.')
            plt.plot(obs, -0.00005 * numpy.ones(obs.shape[0]), "|", color='black')
            plt.legend()


            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors="#D16103", linestyle='--', label='Test stat.')
            plt.vlines(numpy.nanquantile(t_sim, q=1 - signif_lev), ymin=0, ymax=s.dataLim.bounds[3], colors="#293352",
                       linestyle='--', label='95%')
            plt.legend()


        if obs.ndim == 2:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.kdeplot(x=mod[0,:], y=mod[1,:], levels=[0.25,0.50,0.75], colors="#4E84C4", linestyles='--',
                        label='Model')
            sns.kdeplot(x=obs[0,:], y=obs[1,:], levels=[0.25,0.50,0.75], colors="#D16103", label='Obs.')
            plt.scatter(x=obs[0,:], y=obs[1,:], s=2, color='black')
            plt.legend()

            plt.subplot(1, 2, 2)
            s = sns.kdeplot(t_sim, fill=True, color="gray")
            plt.vlines(t, ymin=0, ymax=s.dataLim.bounds[3], colors="#D16103",linestyle='--', label='Test stat.')
            plt.vlines(numpy.nanquantile(t_sim, q=1 - signif_lev), ymin=0, ymax=s.dataLim.bounds[3], colors="#293352",
                       linestyle='--', label='95%')
            plt.legend()

    return pd.DataFrame({'Test': t, 'Bootstrap': t_sim, 'P_value': p_value})



if __name__ == '__main__':
    print('Welcome to ConTEST, check out the github page to learn more about it.')