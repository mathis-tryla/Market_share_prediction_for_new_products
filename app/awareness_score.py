# -*- coding: utf-8 -*-
import numpy as np
import math, os, sys, vaex
import pandas as pd

def get_awareness_score(dataset, ad_hoc):
  worksheet = ad_hoc 
  df = vaex.from_pandas(pd.read_excel(worksheet, sheet_name=0))[1:]

  # Number of competing products in the results of sales
  # Number not equal to 0 
  cmd = "awk -F\";\" \'{print $1}\' " + dataset + " | sort | uniq -c | wc -l > competing_products_number"
  # Check if code is run on MacOS
  if sys.platform == 'darwin':
    cmd = "gawk -F\";\" \'{print $1}\' " + dataset + " | sort | uniq -c | wc -l > competing_products_number"
  os.system(cmd)
  with open('./competing_products_number', 'r') as f:
    competing_product_number = f.read()
  
  #get the row with purchase frequency and convert it to array
  df.rename('A quelle fréquence souhaitez-vous racheter ce produit dans le futur ?', 'purchase_frequency')
  dffrequency=df['purchase_frequency'].values

  # we create a data list list of frequencies according to their numerical value.
  # The value corresponds to the purchase recurrence in 1 month.
  # For example, a purchase every 2 weeks, will be 2.15 per month (1 month = 4.3 weeks)
  data_purchase_frequency={'frequency':  ['2 fois ou plus par semaine.', '1 fois par semaine.',
                                          '1 fois toutes les 2 semaines.', '1 fois par mois.',
                                         'Au plus une fois tous les 2 mois.'],
         'time_per_month': [8.6, 4.3, 2.15, 1, 0.5]
         }
  dfdata = vaex.from_dict(data_purchase_frequency)

  #we we browse the frequency list and we will change the string values into int thanks to the conversion table created before
  #So, if in a line, it is indicated that the product is purchased once a month, we convert it by its recurrence in one month, i.e. 1
  
  df_frequency = [] #For 1 time every week, the recurrence is 4.3 per month so the sentence will be replaced by 4.3 
  for i in range(0,len(dfdata)):
    for j in range(0,len(dffrequency)):
      if dfdata['frequency'].values[i]==dffrequency[j]:
        df_frequency.append(dfdata['time_per_month'].values[i])

  #we calculate the average of the purchase frequency
  avg_frequency = np.mean(df_frequency)

  # formula to have the visibility score 
  # avg_frequency : int , frequency in one month 
  # competing_number-product : int,  number of competing prducts , found in the dataset
  # on the same basis as the GRP
  calcul = avg_frequency / 4 / (1+math.log(int(competing_product_number))) 

  #if we wanted a theoretical value for products, on average, 
  #6 bottles are sold per year or 1 every 2 months. The frequency would have been 0.5 
  #theoritical_frequency = 0.5
  #calcul2 = theoritical_frequency/4/(1+math.log(competing_product_number))  
  print("-- Calculate awareness score DONE")
  return calcul