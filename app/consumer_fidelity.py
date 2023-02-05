# -*- coding: utf-8 -*-
import pandas as pd
import vaex

def get_fidelity_score(ad_hoc):
  worksheet = ad_hoc 
  df = vaex.from_pandas(pd.read_excel(worksheet, sheet_name=0))[1:]

  #get the ad-hoc row that is the number of differents products used from the same product category in one year and convert it to array
  #dfproductsperyear=df.iloc[:, 14].values
  df.rename('Combien achetez vous de produits différents issus de cette catégorie en 1 an?', 'nb_prod_to_buy')
  dfproductsperyear = df['nb_prod_to_buy'].values

  # we calculate the average of the column
  avg = 0
  for i in range(0, len(dfproductsperyear)):
    avg += int(dfproductsperyear[i])
  avg = avg / len(dfproductsperyear)
  
  fidelity_score = 1 / avg 
  print("-- Calculate consumers fidelity score DONE")
  return fidelity_score