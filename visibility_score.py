# -*- coding: utf-8 -*-
"""visibility_score.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PuNPjpvUOV3EJKojeIuJx8G4FjtKY0r1
"""

def get_visibility():

  from google.colab import auth
  import gspread
  from google.auth import default
  import numpy as np
  import math

  #autenticating to google
  auth.authenticate_user()
  creds, _ = default()
  gc = gspread.authorize(creds)
  import pandas as pd

  # Number of competing products in the results of sales
  # Number not equal to 0 
  competing_product_number = 15379

  #defining the ad-hoc worksheet 
  worksheet = gc.open('Ranking Ad-hoc questionnaire (réponses)').sheet1
  #get_all_values gives a list of rows
  rows = worksheet.get_all_values()
  #Convert values to a DataFrame 
  df = pd.DataFrame(rows)
  
  # we delete the first row(titles)
  df.drop(df.head(1).index, inplace=True)
  #get the row with purchase frequency and convert it to array
  dffrequency=df[11].to_numpy()

  # we create a data list list of frequencies according to their numerical value.
  # The value corresponds to the purchase recurrence in 1 month.
  # For example, a purchase every 2 weeks, will be 2.15 per month (1 month = 4.3 weeks)
  data_purchase_frequency={'frequency':  ['2 fois ou plus par semaine.', '1 fois par semaine.',
                                          '1 fois toutes les 2 semaines.', '1 fois par mois.',
                                         'Au plus une fois tous les 2 mois.'],
         'time_per_month': [8.6, 4.3, 2.15,1,0.5]
         }
  dfdata = pd.DataFrame(data_purchase_frequency, columns=['frequency', 'time_per_month'])

  #we we browse the frequency list and we will change the string values into int thanks to the conversion table created before
  #So, if in a line, it is indicated that the product is purchased once a month, we convert it by its recurrence in one month, i.e. 1
  #For 1 time every week, the recurrence is 4.3 per month so the sentence will be replaced by 4.3 
  for i in range(0,len(dfdata)):
    for j in range(0,len(dffrequency)):
      if dfdata['frequency'][i]==dffrequency[j]:
        dffrequency[j]=dfdata['time_per_month'][i]

  #we convert array to dataframe
  dffrequency = pd.DataFrame(dffrequency)

  #we calculate the average of the purchase frequency
  avg_frequency = np.mean(dffrequency)

  # formula to have the visibility score 
  # avg_frequency : int , frequency in one month 
  # competing_number-product : int,  number of competing prducts , found in the shampoo dataset
  # on the same basis as the GRP
  calcul = avg_frequency /4/(1+math.log(competing_product_number)) 

  #if we wanted a theoretical value for shampoos, on average, 
  #6 bottles are sold per year or 1 every 2 months. The frequency would have been 0.5 
  #theoritical_frequency = 0.5
  #calcul2 = theoritical_frequency/4/(1+math.log(competing_product_number)) 
  
  return calcul

get_visibility()
