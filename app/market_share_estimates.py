import sys, os, vaex
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pandas as pd

# Get n products barcodes which are competitors of the new product
def find_n_competing_products(df, nb_competing_products):
  n_products_barcodes = []
  cpt = 0
  for barcode in df.barcode.unique():
    n_products_barcodes.append(barcode)
    cpt += 1
    if cpt == nb_competing_products:
      print(f"-- Find {nb_competing_products} products DONE")
      return n_products_barcodes

"""
Get the sales numbers of the n competitors by their barcode
and add the new product dataframe row
"""
def get_sales_numbers_for_n_products(df, n_barcodes):
  n_competing_barcodes = []
  for barcode in n_barcodes:
    if barcode != 'barcode new product':
      n_competing_barcodes.append(barcode)

  n_products_dict = {}
  sales_numbers = []

  for barcode in n_competing_barcodes:
    sales_number = df[(df['barcode'] == barcode)].sales_number.values.sum()
    sales_numbers.append(sales_number)

  n_competing_barcodes.append("barcode new product")
  n_products_dict["barcode"] = n_competing_barcodes
  sales_numbers.append(0)
  n_products_dict["sales_number"] = sales_numbers 
  print(f"-- Get sales numbers for {len(n_barcodes)} products DONE")
  return vaex.from_dict(n_products_dict)

# Calculate the market shares of the n competitors by retrieving their sales numbers
def get_competing_products_mshares(dataframe):
  total_sales_number = dataframe['sales_number'].sum()
  market_shares_list = [] 
  if total_sales_number > 0:  
    for sn in dataframe.sales_number.values:
      market_share = (sn/total_sales_number) * 100
      market_shares_list.append(market_share)
  dataframe['market_share'] = np.array(market_shares_list)
  print(f"-- Get competing products market shares DONE")
  return dataframe

"""
Get the ad-hoc values for the six products 
(n competitors and the new product)
"""
def extract_adhoc(ad_hoc, nb_competing_products):
  worksheet = ad_hoc 
  df = vaex.from_pandas(pd.read_excel(worksheet, sheet_name=0))
  dfprods = []
  for n in range(1, nb_competing_products+1):
    df.rename(f'Classez le produit étudié et ces 6 produits en fonction de votre volonté d’achat (ranking) entre 1 et 7 [prod{n}]', f'prod{n}')
    dfprods.append(df[f'prod{n}'])
  df.rename('Classez le produit étudié et ces 6 produits en fonction de votre volonté d’achat (ranking) entre 1 et 7 [prod_test]', 'prod_test')
  dfprodTEST = df['prod_test']
  dfprods.append(dfprodTEST)

  ranking_products = []
  dict_ranking_products = {}

  for product in dfprods:
    ranking_score_product = 0
    # Get the number of occurences purcentage
    # from 1 to 6
    for number in range(1, product.nunique()+1):
      cpt = 0
      for elt in product.values:
        if elt == number:
          cpt+=1 
      purcentage = cpt / len(product.values)
      # Score formula 
      # If you have 7 products , from 7 points if you are first to 1 point if you are seventh
      score = product.nunique() + 1 - number
      ranking_score_number = purcentage*score
      ranking_score_product += float(ranking_score_number)
    ranking_products.append(ranking_score_product)
  dict_ranking_products['ranking_score'] = ranking_products
  print(f"-- Extract ad-hoc values DONE")
  return vaex.from_dict(dict_ranking_products)

# Apply the polynomial regression depending on the degree of the polynomial
def eval_regression(degree, X, y):
    # Transform features to include higher-order terms
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Fit linear regression model
    reg = LinearRegression()
    model = reg.fit(X_poly, y)

    # Predict using the model
    y_predicted = reg.predict(X_poly)

    # Calculate mean squared error
    rmse = mean_squared_error(y, y_predicted, squared=False)
    return rmse,model,y_predicted

# Get the degree of the polynomial regression regarding the smallest Root Mean Squared Error
def get_convenient_regression_model(min_degree, max_degree, X, y):
  convenient_degree = sys.maxsize
  convenient_y_predicted = None
  min_rmse = sys.maxsize
  min_model = ''
  if min_degree < max_degree:
    for degree in range(min_degree, max_degree+1):
      rmse,model,y_predicted = eval_regression(degree, X, y)
      if rmse < min_rmse:
        convenient_degree = degree
        min_rmse = rmse
        min_model = model
        convenient_y_predicted = y_predicted
  print(f"-- Get convenient regression model DONE")
  return convenient_degree,min_rmse,min_model,convenient_y_predicted

"""
Predict the market share of the six products from the ranking scores
retrieved from the extract_adhoc() method 
"""
def get_new_mshares(dataframe):  
  X, y = dataframe["ranking_score"].values[:-1].reshape(-1,1), dataframe["market_share"].values[:-1]
  x=X[:,0]
  
  for score in dataframe.ranking_score.values: 
    if score is None:
      return pd.DataFrame()

  # Get the convenient degree of the polynmial regression
  degree,rmse,_,y_predicted = get_convenient_regression_model(1, 10, x.reshape(-1,1), y)
  new_product_ranking_score = dataframe["ranking_score"].values[-1]
  print(f"-- Convenient degree = {degree} DONE")
  
  # Get the new product market share
  mymodel = np.poly1d(np.polyfit(x, y, degree))
  new_product_market_share = mymodel(new_product_ranking_score)
  print("-- Predict the new product market share DONE")

  # Create the dataframe of the six products
  six_products_dict = {}
  six_products_dict["barcode"] = dataframe["barcode"]
  six_products_dict["sales_number"] = dataframe["sales_number"]
  six_products_dict["ranking_score"] = dataframe["ranking_score"]
  six_products_dict["old_market_share"] = dataframe["market_share"]
  y_predicted = np.append(y_predicted, new_product_market_share)
  
  # Scale the target column to have a sum of 100
  dataframe["market_share"] = (y_predicted/y_predicted.sum()) * 100
  six_products_dict["new_market_share"] = y_predicted
  
  print(f"-- Get new market shares DONE")
  return six_products_dict, rmse

# Main method to run in order to output the new product market share
def get_np_market_share_max(df, ad_hoc, nb_competing_products):
  if os.path.isfile(ad_hoc):
    competing_products_barcodes = find_n_competing_products(df, nb_competing_products)
    df = df.dropna()
    df_ranking_scores = extract_adhoc(ad_hoc, nb_competing_products)
    df_n_products = get_sales_numbers_for_n_products(df, competing_products_barcodes)
    if df_n_products['sales_number'].sum() > 0:
      df_n_products_mshares = get_competing_products_mshares(df_n_products)
      df_n_products_mshares["ranking_score"] = np.array(df_ranking_scores.ranking_score.values)
      output_df, rmse = get_new_mshares(df_n_products_mshares)
      market_share_new_product = output_df['new_market_share'][-1]
      return market_share_new_product, rmse
  sys.exit(-1)
