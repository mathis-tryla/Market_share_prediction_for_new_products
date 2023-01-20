import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

targeted_nb_of_contacts = int(sys.argv[1])
reach = float(sys.argv[2])
frequency = float(sys.argv[3])

print("Percentage of population with " + str(targeted_nb_of_contacts) + " or more contact with advertising, for " + str(frequency) + " airings with " + str(round(reach,2)) + " mean reach")
avgPopHit = stats.nbinom.cdf(frequency-targeted_nb_of_contacts, targeted_nb_of_contacts, reach)
print(avgPopHit)

x = np.arange(stats.nbinom.ppf(0.01, targeted_nb_of_contacts, reach),stats.nbinom.ppf(avgPopHit+0.01, targeted_nb_of_contacts, reach))

fig, ax = plt.subplots(1, 1)
ax.plot(x, stats.nbinom.pmf(x, targeted_nb_of_contacts, reach), 'bo', ms=7, label='nbinom pmf')
ax.vlines(x, 0, stats.nbinom.pmf(x, targeted_nb_of_contacts, reach), colors='b', lw=6, alpha=0.5)

plt.show(block=True)
