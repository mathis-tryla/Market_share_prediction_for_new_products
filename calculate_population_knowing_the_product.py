# How to run this script :
# 1: python3 calculate_population_knowing_the_product.py targeted_nb_of_contacts(int) reach(float) frequency(float)
# Ex : python3 calculate_population_knowing_the_product.py 3 0.25 10

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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
    df = pd.DataFrame(data=data)
    df.set_index('week', inplace=True)
    df.to_csv("population_knowing_the_product.csv")

def get_population_knowing_product(targeted_nb_of_contacts, reach, frequency, length_of_campaign):
    if targeted_nb_of_contacts is not None and reach is not None and frequency is not None and length_of_campaign is not None:
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

        return data
    return None
