from CONTEST import *

df = pd.read_csv("pythonDataCheck.csv", index_col=0)

# Invoking the R function and getting the result
Test,Boot,Pvalue = ConTEST_reg(df,K=10)