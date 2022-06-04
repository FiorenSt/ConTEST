####REGRESSION DENSITY

ConTEST_dens=function(mod,obs,K=100,seed=10,signif_lev=0.05){
  
  ###OBSERVATION FROM REAL MODEL
  set.seed(seed)
  
  ###MODEL IN EXAM (OBTAINED AS KDE FROM SIMULATED POINTS)
  g =  kde(mod,binned=F) #,eval.points = grid
  
  ####DENSITY TO ESTIMATE THE OBSERVATIONAL PART
  f= kde(obs, binned=F)  
  
  
  ####APPROXIMATE DISTANCE WITH MCMC
  J=100
  
  ##simulate from g
  x_g=rkde(J,g)
  
  d_g=dkde(x_g,g)
  d_f=dkde(x_g,f)
  
  d_kull =  sum( abs(d_f/d_g -1)) /J


  # - simulate K dataset 
  d_sim = NULL
  
  
  for( k in 1:K){
    simulation = rkde(dim(obs)[1],g)
    f_sim=kde(simulation,binned=F)
    d_f=dkde(x_g,f_sim)
    #d_g=dkde(x_g,g)
    d_sim[k] =  sum( abs(d_f/d_g -1)) /J
    }
  
  
  #Hypothesis testing
  p_value = min( mean(d_sim <= d_kull), mean(d_sim >= d_kull))
  
  return(list(Test_stat=d_kull, Boot_stat=d_sim, P_value=p_value))
}




#obs=mvtnorm::rmvnorm(n=100, mean=c(5,5),sigma=matrix(c(2,0.2,0.2,3),2,2))
#mod=mvtnorm::rmvnorm(n=100, mean=c(5,5),sigma=matrix(c(2,0.2,0.2,3),2,2))
#df=data.frame(mod = mod)
#df2=data.frame(obs = obs)
#write.csv(df, 'pythonDataCheck_mod.csv')
#write.csv(df2, 'pythonDataCheck_obs.csv')



