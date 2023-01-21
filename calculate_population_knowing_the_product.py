# How to run this script :
# 1: python3 calculate_population_knowing_the_product.py targeted_nb_of_contacts(int) reach(float) frequency(float)
# Ex : python3 calculate_population_knowing_the_product.py 3 0.25 10

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Show the plot of the distribution of population knowing the product
def show_plot(targeted_nb_of_contacts, reach, avgPopHit):

    x = np.arange(stats.nbinom.ppf(0.01, targeted_nb_of_contacts, reach),
                  stats.nbinom.ppf(avgPopHit + 0.01, targeted_nb_of_contacts, reach))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, stats.nbinom.pmf(x, targeted_nb_of_contacts, reach), 'bo', ms=7, label='nbinom pmf')
    ax.vlines(x, 0, stats.nbinom.pmf(x, targeted_nb_of_contacts, reach), colors='b', lw=6, alpha=0.5)

    plt.show(block=True)
    return

def main():
    if len(sys.argv) > 1:
        targeted_nb_of_contacts = int(sys.argv[1])
        reach = float(sys.argv[2])
        frequency = float(sys.argv[3])

        print("Percentage of population with " + str(targeted_nb_of_contacts) + " or more contact with advertising, for " + str(frequency) + " airings with " + str(round(reach, 2)) + " mean reach")
        avgPopHit = stats.nbinom.cdf(frequency-targeted_nb_of_contacts, targeted_nb_of_contacts, reach)
        print(avgPopHit)

        show_plot(targeted_nb_of_contacts, reach, avgPopHit)
        return avgPopHit
    else:
        print("wrong number of arguments")
        return 0


start_time = time.time()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- Calculate population knowing the product DONE")
print(f"-- Output = {output} DONE")
print(f"-- {final_time} seconds--")
