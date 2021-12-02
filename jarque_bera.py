# Sometimes mean and variance are not enough to describe a distribution.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera

# Plot a normal distribution with mean = 0 and standard deviation = 2
xs = np.linspace(-6,6,300)
normal = stats.norm.pdf(xs)
plt.subplot(211)
plt.plot(xs, normal)

# Generate x-values for which we will plot the distribution
xs2 = np.linspace(stats.lognorm.ppf(0.01, 0.7, loc=-0.1), stats.lognorm.ppf(0.99, 0.7, loc=-0.1), 150) 
# Negatively skewed distribution
lognormal = stats.lognorm.pdf(xs2, 0.7)
plt.subplot(212)
plt.plot(xs2, lognormal, label='Skew > 0')
plt.plot(xs2, lognormal[::-1], label='Skew < 0')
plt.legend() 
plt.show()

# Some sample distributions 
# Kurtosis attempts to measure the shape of the deviation from the mean.  
# Generally, it describes how peaked a distribution is compared to the normal distribution.
# 
# All normal distribution have a kurtosis of 3.

plt.figure
plt.plot(xs,stats.laplace.pdf(xs), label='Leptokurtic')
print 'Excess kurtosis of leptokurtic distribution:', (stats.laplace.stats(moments='k'))
plt.plot(xs, normal, label='Mesokurtic (normal)')
print 'Excess kurtosis of mesokurtic distribution:', (stats.norm.stats(moments='k'))
plt.plot(xs,stats.cosine.pdf(xs), label='Playkurtic')
print 'Excess kurtosis of platykurtic distribution:', (stats.cosine.stats(moments='k'))
plt.legend()
plt.show()


N = 1000
M = 1000

pvalues = np.ndarray( (N) )

for i in range(N):
  # Draw M samples from a normal distribution
  X = np.random.normal(0, 1, M)
  _, pvalue, _, _ = jarque_bera(X)
  pvalues[i] = pvalue

# count number of pvalues below our default 0.05 cutoff
num_significant = len(pvalues[pvalues < 0.05])

print float(num_significant) / N



