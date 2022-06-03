import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects


def load_data(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_r = robjects.conversion.py2rpy(df)
    return(df_r)


if __name__ == "__main__":

    df = pd.read_csv("pythonDataCheck.csv", index_col=0)
    data=load_data(df)

    # Invoking the R function and getting the result
