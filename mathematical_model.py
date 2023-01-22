# How to run this script :
# 1: python3 mathematical_model.py visibility_score(float) fidelity_score(float) monthly_growth_rate(file.csv) market_share_max(float)
# Ex: python3 mathematical_model.py 0.1 0.0875 monthly_growth_rate.csv 25

import math, os, sys, time, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas

from calculate_population_knowing_the_product import get_population_knowing_product
from market_share_estimates import get_np_market_share_max
from shampoos_seasonality_multipliers import get_shampoos_seasonality_multipliers
from consumer_fidelity import get_fidelity_score
from visibility_score import get_visibility_score

def show_model_simulation(entity_per_week, type):
    x = np.array(range(26))
    y = entity_per_week

    plt.plot(x, y, "r", label="Predicted")
    plt.title("The evolution of the market share of the new product")
    plt.xlabel("time (week)")
    if type == "ms":
        plt.ylabel("market share (%)")
    elif type == "sn":
        plt.ylabel("sales number")
    plt.legend(loc="lower right")
    plt.show()

def export_to_csv(data_points, type):
    df_points = pandas.DataFrame(data=data_points)
    df_points.set_index('week', inplace=True)
    csv_file = "points.csv"
    if type == "ms":
        csv_file = "points_market_shares.csv"
    elif type == "sn":
        csv_file = "points_sales_numbers.csv"
    df_points.to_csv(csv_file)

def main():
    dataset = input("Shampoos dataset file: ")
    remove_bad_lines_file = input("Remove bad lines file: ")
    ad_hoc = input("Ad-hoc file: ")
    targeted_nb_of_contacts = int(input("Targeted nb of contacts: "))
    reach = float(input("Reach: "))
    frequency = float(input("Frequency: "))
    length_of_campaign = int(input("Length of campaign: "))
    DV = float(input("DV (between 0 and 1): "))

    if not os.path.isfile(dataset) or not os.path.isfile(remove_bad_lines_file) or not os.path.isfile(ad_hoc):
        sys.exit("Dataset, remove bad lines file or ad-hoc file doesn't exist!")

    population_knowing_the_product = get_population_knowing_product(targeted_nb_of_contacts, reach, frequency, length_of_campaign)
    shampoos_seasonality_multipliers, total_sales_numbers = get_shampoos_seasonality_multipliers(dataset, remove_bad_lines_file)
    visibility_score = float(get_visibility_score(dataset, ad_hoc))
    fidelity_score = float(get_fidelity_score(ad_hoc))
    market_share_max = float(get_np_market_share_max(dataset, ad_hoc))
    displacement = 1
    global_growth_rate = visibility_score + fidelity_score

    # one point for each week of the 1st semester on the market
    market_share_per_week, sales_number_per_week = [], []
    for week in range(26):
        print(week)
        # Gompertz curve equation
        print(f"population_knowing_the_product[predicted][week]={population_knowing_the_product['predicted'][week]}")
        market_share_curr_week = (population_knowing_the_product["predicted"][week]/100) * market_share_max * math.exp(-displacement * math.exp(-global_growth_rate * week))
        market_share_per_week.append(market_share_curr_week)
    # Plot market shares and sales number of the new product per week
    # ms = market share
    # sn = sales number
    show_model_simulation(market_share_per_week, "ms")

    for _, seasonality in shampoos_seasonality_multipliers.items(): 
        # Sales numbers per week
        # Vente(i) = part de marché(i) * volume du marché(i) * DV * coeff de saisonnalité(i)
        market_volume_curr = seasonality * total_sales_numbers
        sales_number_curr_week = market_share_curr_week * market_volume_curr * DV * seasonality 
        sales_number_per_week.append(sales_number_curr_week)
    show_model_simulation(sales_number_per_week[:26], "sn")

    # export points to csv
    data_points_ms = {'week': range(26), 'predicted': market_share_per_week}
    data_points_sn = {'week': range(26), 'predicted': sales_number_per_week[:26]}
    export_to_csv(data_points_ms, "ms")
    export_to_csv(data_points_sn, "sn")

    return "csv file(s) created"


start_time = time.time()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- Calculate population knowing the product DONE")
print(f"-- Output = {output} DONE")
print(f"-- {final_time} seconds --")
