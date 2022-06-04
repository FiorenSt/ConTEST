####REGRESSION FUNCTION
#library(np)

ConTEST_reg=function(y_mod,x_obs,y_obs,uncertainties,K=10,seed=10,signif_lev=0.05){

  # - nonparametric regression on the data
  N=length(x_obs)
  bw.fixed <- npregbw(y_obs ~ x_obs,
                      regtype = "ll",
                      bwmethod = "cv.aic",
                      gradients = FALSE,
                      bwtype= 'fixed',
                      nmulti=1)
  
  model.fixed = npreg(bw.fixed)
  
  # - distance calculation
  est_fixed=model.fixed$mean
  unc_fixed=model.fixed$merr
  
  ##Test statistic
  d_fixed =  sqrt( sum(((est_fixed-y_mod)/unc_fixed)^2)/N)
  
  # - simulate K dataset 
  set.seed(seed)

  sim=matrix(NA,N,K)
  
  for(k in 1:K){
    for(i in 1:N){
      sim[i,k] = y_mod[i] + rnorm(1,0,sd=uncertainties[i])
    }
  }
  
  # - nonparametric regression on the simualted data
  est_sim = matrix(NA,N,K)
  est_sim_unc = matrix(NA,N,K)
  

  for( k in 1:K){
    simulation = data.frame(x_sim=x_obs,y_sim=sim[,k])
    
    bw.sim <- npregbw(y_sim ~ x_sim,
                      regtype = "ll",
                      bwmethod = "cv.aic",
                      gradients = F,
                      data = simulation,
                      bwtype= 'fixed',
                      nmulti=1)
    
    model.sim = npreg(bw.sim)
    est_sim[,k]=model.sim$mean
    est_sim_unc[,k]=model.sim$merr
    #print(k)
  }
  
  #Hypothesis testing
  d_sim =  sqrt( colSums(((est_sim-y_mod)/est_sim_unc)^2)/N)
  p_value = min( mean(d_sim <= d_fixed), mean(d_sim >= d_fixed))

  return(list(Test_stat=d_fixed, Boot_stat=d_sim, P_value=p_value))
}



#output=ConTEST_reg(y_mod = MODEL,x_obs = X, y_obs = OBS, uncertainties = ERR_OBS,K=10)
#df=data.frame(y_mod = MODEL,x_obs = X, y_obs = OBS, uncertainties = ERR_OBS)
#write.csv(df, 'pythonDataCheck.csv')
#output
