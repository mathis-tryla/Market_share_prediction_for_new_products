import matplotlib.pyplot as plt
import sys, time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from io import StringIO
from scipy.special import logsumexp

def create_dataframe():
  headers = ['timestamp','week','barcodes','type','segment','category','description','weight','sales_numbers','price','sales_value','discounts']
  return pd.read_csv(sys.argv[1], sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore")

def sort_data_per_barcode(df):
  return df.sort_values('barcodes')

def find_five_products(df):
  five_products_barcodes = []
  for barcode in df.barcodes.unique():
    five_products_barcodes.append(barcode)
  return five_products_barcodes[:5]

def find_unique_weeks(df):
  return df["week"].unique()

def get_sales_numbers_for_five_shampoos(df, five_barcodes, week):
  five_new_barcodes = []
  for barcode in five_barcodes:
    if barcode != 'barcode new product':
      five_new_barcodes.append(barcode)

  five_products_dict = {}
  timestamps = []
  categories = []
  descriptions = []
  sales_numbers = []

  for barcode in five_new_barcodes:
    sales_numbers_arr = df.loc[(df['barcodes'] == barcode) & (df['week'] == week), ['barcodes','sales_numbers']].sales_numbers.values
    sales_number = 0
    for sn in sales_numbers_arr:
      sales_number += sn
    sales_numbers.append(sales_number)

  five_new_barcodes.append("barcode new product")
  five_products_dict["barcodes"] = five_new_barcodes
  sales_numbers.append(0)
  five_products_dict["sales_numbers"] = sales_numbers 
  return pd.DataFrame(five_products_dict)

def get_competing_products_mshares(dataframe):
  total_sales_number = dataframe['sales_numbers'].sum()
  market_shares_list = [] 
  print(f"total_sales_number={total_sales_number}")
  if total_sales_number > 0:  
    for sn in dataframe['sales_numbers']:
      market_share = (sn/total_sales_number) * 100
      market_shares_list.append(round(market_share, 1))
  dataframe['market_shares'] = market_shares_list
  return dataframe

def extract_adhoc():
  rankingscores = []
  worksheet = sys.argv[2] 
  df = pd.read_excel(worksheet, sheet_name=0)
  dfprods = []
  dfprod1 = df.iloc[:, 5]
  dfprod2 = df.iloc[:, 6]
  dfprod3 = df.iloc[:, 7]
  dfprod4 = df.iloc[:, 8]
  dfprod5 = df.iloc[:, 9]
  dfprodTEST = df.iloc[:, 10]

  dfprods.append(dfprod1);dfprods.append(dfprod2);dfprods.append(dfprod3)
  dfprods.append(dfprod4);dfprods.append(dfprod5);dfprods.append(dfprodTEST)
  ranking_products = []
  dict_ranking_products = {}

  for product in dfprods:
    ranking_score_product = 0
    # Get the number of occurences purcentage
    # from 1 to 6
    for number in range(1, product.nunique()+1):
      purcentage = product.loc[product == number].value_counts() / len(product)
      #score formula 
      #if you have 7 products , from 7 points if you are first to 1 point if you are seventh
      score = product.nunique() + 1 - number
      ranking_score_number = purcentage*score
      if len(ranking_score_number.values) > 0:
        ranking_score_product += float(ranking_score_number)
    ranking_products.append(ranking_score_product)
  dict_ranking_products['ranking_scores'] = ranking_products
  return pd.DataFrame(dict_ranking_products)

def eval_regression(degree, X, y, test_size, random_state):
    # Transform features to include higher-order terms
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Split data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=random_state)

    # Fit linear regression model
    reg = LinearRegression()
    model = reg.fit(X_poly, y)

    # Predict using the model
    y_predicted = reg.predict(X_poly)

    # Calculate mean squared error
    rmse = mean_squared_error(y, y_predicted, squared=False)
    return rmse,model,y_predicted

def get_convenient_regression_model(min_degree, max_degree, X, y, test_size, random_state):
  convenient_degree = sys.maxsize
  convenient_X = None
  convenient_y_predicted = None
  min_rmse = sys.maxsize
  min_model = ''
  final_score = -1
  if min_degree < max_degree:
    for degree in range(min_degree, max_degree+1):
      rmse,model,y_predicted = eval_regression(degree, X, y, test_size, random_state)
      if rmse < min_rmse:
        convenient_degree = degree
        min_rmse = rmse
        min_model = model
        convenient_y_predicted = y_predicted
  return convenient_degree,min_rmse,min_model,convenient_y_predicted

def get_new_mshares(dataframe):  
  # Scale the target column to have a sum of 100
  #dataframe["market_shares"] = (dataframe["market_shares"]/dataframe["market_shares"].sum()) * 100

  print(dataframe)

  # Predict the market share of the new market shares from the ranking scores 
  X, y = dataframe["ranking_scores"].iloc[:-1].values.reshape(-1,1), dataframe["market_shares"].iloc[:-1]
  x=X[:,0]

  if dataframe["ranking_scores"].isnull().values.any():
    return pd.DataFrame()

  degree,rmse,model,y_predicted = get_convenient_regression_model(1, 10, x.reshape(-1,1), y, 0.3, 42)
  print(f"GET market_shares FROM ranking_scores = {degree}")
  print(f"y_predicted = {y_predicted}")
  #new_product_market_share = 100 - y_predicted.sum()
  #new_product_market_share = y_predicted[-1]
  #print(f"new_product_market_share={new_product_market_share}")
  new_product_ranking_score = dataframe["ranking_scores"].iloc[-1]
  mymodel = np.poly1d(np.polyfit(x, y, degree))
  new_product_market_share = mymodel(new_product_ranking_score)
  print(f"new_product_market_share={new_product_market_share}")
  print("-- predict the new product market share DONE")

  six_products_dict = {}
  six_products_dict["barcodes"] = dataframe["barcodes"]
  six_products_dict["sales_numbers"] = dataframe["sales_numbers"]
  six_products_dict["ranking_scores"] = dataframe["ranking_scores"]
  six_products_dict["old_market_shares"] = dataframe["market_shares"]
  y_predicted = np.append(y_predicted, round(new_product_market_share,2))
  # Scale the target column to have a sum of 100
  dataframe["market_shares"] = (y_predicted/y_predicted.sum()) * 100
  six_products_dict["new_market_shares"] = dataframe["market_shares"]
  df_six_products = pd.DataFrame(six_products_dict)
  print(df_six_products)
  # Predict the sales number of the new product from the new market shares 
  """X, y = df_six_products["new_market_shares"].iloc[:-1].values.reshape(-1, 1), df_six_products["sales_numbers"].iloc[:-1]
  x=X[:,0]
  degree,rmse,model,y_predicted = get_convenient_regression_model(1, 10, x.reshape(-1,1), y, 0.3, 42)
  print(f"GET sales_numbers FROM new_market_shares = {degree}")
  mymodel = np.poly1d(np.polyfit(x, y, degree))
  new_product_market_share = df_six_products["new_market_shares"].iloc[-1]
  estimated_new_product_sales_number = mymodel(new_product_market_share)

  #plot_polynomial(new_product_market_share, estimated_new_product_sales_number, mymodel, 0, 100)

  six_products_dict["new_sales_numbers"] = y_predicted
  six_products_dict["new_sales_numbers"] = np.append(six_products_dict["new_sales_numbers"], estimated_new_product_sales_number)"""
  return df_six_products

def plot_polynomial(x, y, polynomial, start, stop):
  xp = np.linspace(start, stop, 100)
  _ = plt.plot(x, y, '.', xp, polynomial(xp))
  #plt.ylim(-100000,40000)
  plt.show()

def main():
  final_market_shares = {}
  df = create_dataframe()
  df = sort_data_per_barcode(df)
  five_products_barcodes = find_five_products(df)
  unique_weeks = find_unique_weeks(df)
  df = df.dropna()
  df_ranking_scores = extract_adhoc()
  outputs_df = {}
  five_bc = []
  for week in unique_weeks:
      print(f"week={week}")
      df_five_products = get_sales_numbers_for_five_shampoos(df, five_products_barcodes, week)
      if df_five_products['sales_numbers'].sum() > 0:
        df_five_products_mshares = get_competing_products_mshares(df_five_products)
        df_five_products_mshares.insert(2, "ranking_scores", pd.Series(df_ranking_scores['ranking_scores'].values), allow_duplicates=True)
        outputs_df[week] = get_new_mshares(df_five_products_mshares)
        final_market_shares[week] = outputs_df[week]['new_market_shares'].iloc[-1] 
      else:
        outputs_df[week] = None
  return final_market_shares
  
start_time = time.time()
outputs_df = main()
end_time = time.time()
final_time = end_time-start_time
print(f"-- {final_time} seconds --")
print(outputs_df)