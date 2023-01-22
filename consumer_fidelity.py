# -*- coding: utf-8 -*-
import pandas as pd

def get_fidelity_score(ad_hoc):
  worksheet = ad_hoc 
  df = pd.read_excel(worksheet, sheet_name=0)

  # we delete the first row(titles)
  df.drop(df.head(1).index, inplace=True)
  #get the ad-hoc row that is the number of differents products used from the same product category in one year and convert it to array
  dfproductsperyear=df.iloc[:, 14].values
  # dfproductsperyear=df[df.keys()[15]].to_numpy()

  # we calculate the average of the column
  avg=0
  for i in range(0,len(dfproductsperyear)):
    avg+=int(dfproductsperyear[i])
  avg = avg/len(dfproductsperyear)
  
  fidelity_score = 1/avg 

  return fidelity_score
