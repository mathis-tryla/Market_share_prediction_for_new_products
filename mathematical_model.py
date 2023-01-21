# How to run this script :
# 1: python3 mathematical_model.py visibility_score(float) fidelity_score(float) monthly_growth_rate(file.csv) market_share_max(float) market_share_max_delta(float)
# Ex: python3 mathematical_model.py 0.1 0.0875 monthly_growth_rate.csv 25 0.2

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

import pandas as pandas


def show_model_simulation(market_share_per_week, market_share_max_delta):
    x = np.array(range(26))
    y = market_share_per_week
    y_minus_delta = [ms * (1 - market_share_max_delta) for ms in y]
    y_plus_delta = [ms * (1 + market_share_max_delta) for ms in y]

    plt.plot(x, y, "r", label="Predicted")
    plt.plot(x, y_minus_delta, "b", label="floor")
    plt.plot(x, y_plus_delta, "b", label="ceiling")
    plt.title("The evolution of the market share of the new product")
    plt.xlabel("time (week)")
    plt.ylabel("market share (%)")
    plt.legend(loc="lower right")
    plt.show()

def export_to_csv(data_points):
    df_points = pandas.DataFrame(data=data_points)
    df_points.set_index('week', inplace=True)
    df_points.to_csv("points.csv")

def main():
    if len(sys.argv) == 6:
        visibility_score = float(sys.argv[1])
        fidelity_score = float(sys.argv[2])
        # monthly_growth_rate = open(sys.argv[3],'r')
        market_share_max = float(sys.argv[4])
        market_share_max_delta = float(sys.argv[5])
        displacement = 1
        global_growth_rate = visibility_score + fidelity_score

        # one point for each week of the 1st semester on the market
        market_share_per_week = []
        for week in range(26):
            print(week)
            # Gompertz curve equation
            market_share_curr_week = market_share_max * math.exp(-displacement * math.exp(-global_growth_rate * week))
            market_share_per_week.append(market_share_curr_week)
        show_model_simulation(market_share_per_week, market_share_max_delta)

        # export points to csv
        data_points = {'week': range(26), 'floor': [ms * (1 - market_share_max_delta) for ms in market_share_per_week], 'predicted': market_share_per_week, 'ceiling': [ms * (1 + market_share_max_delta) for ms in market_share_per_week]}
        export_to_csv(data_points)

        return "points.csv created"
    else:
        return "wrong number of arguments"


start_time = time.time()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- Calculate population knowing the product DONE")
print(f"-- Output = {output} DONE")
print(f"-- {final_time} seconds--")
