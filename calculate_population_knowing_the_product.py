# How to run this script :
# 1: python3 calculate_population_knowing_the_product.py targeted_nb_of_contacts(int) reach(float) airings_per_week(float) length_of_campaign_in_week(int)
# Ex : python3 calculate_population_knowing_the_product.py 3 0.25 2 10

import time
import matplotlib.pyplot as plt
import pandas
from scipy import stats
import sys


# Show the evolution of the population knowing the product
def show_model_simulation(data):
    x = data['week']
    y = data['predicted']

    plt.plot(x, y, "r", label="Predicted")
    plt.title("The evolution of the population knowing the new product")
    plt.xlabel("time (week)")
    plt.ylabel("population knowing the new product (%)")
    plt.legend(loc="lower right")
    plt.show()


# Export the points to csv
def export_to_csv(data):
    df = pandas.DataFrame(data=data)
    df.set_index('week', inplace=True)
    df.to_csv("population_knowing_the_product.csv")


def main():
    if len(sys.argv) == 5:
        targeted_nb_of_contacts = int(sys.argv[1])
        reach = float(sys.argv[2])
        frequency = float(sys.argv[3])
        length_of_campaign = int(sys.argv[4])
        if length_of_campaign > 26:
            length_of_campaign = 26

        pop_hit_per_week = []
        # Diffusion of the campaign
        for week in range(length_of_campaign):
            airings_after_curr_week = frequency*(week+2)
            pop_hit_curr_week = 100 * stats.nbinom.cdf(airings_after_curr_week - targeted_nb_of_contacts, targeted_nb_of_contacts, reach)
            pop_hit_per_week.append(pop_hit_curr_week)
        # Fill awareness after the campaign
        for week in range(26-length_of_campaign):
            pop_hit_per_week.append(pop_hit_per_week[length_of_campaign-1])

        data = {'week': range(26), 'predicted': pop_hit_per_week}
        export_to_csv(data)

        show_model_simulation(data)

        return "population_knowing_the_product.csv created"
    else:
        return "wrong number of arguments"


start_time = time.time()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- Calculate population knowing the product DONE")
print(f"-- Output = {output} DONE")
print(f"-- {final_time} seconds--")
