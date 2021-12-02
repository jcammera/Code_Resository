import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# Example of ranking data
d1 = [10, 9, 5, 7, 5]
print 'Raw data: ', d1
print 'Ranking: ', list(stats.rankdata(d1, method='average'))

n = 100
def compare_correlation_and_spearman_rank(n, noise):
  X = np.random.poisson(size=n)
  Y = np.exp(X) + noise * np.random.normal(size=n)

  Xrank = stats.rankdata(X, method='average')
  # n-2 is the second to last element
  Yrank = stats.rankdata(Y, method='average')

  diffs = Xrank - Yrank # order doesn't matter since we'll be squaring these values
  r_s = 1 - 6*sum(diffs*diffs)/(n*(n**2 - 1))
  c_c = np.corrcoef(X,Y)[0][1]

  return r_s, c_c


experiments = 1000
spearman_dist = np.ndarray(experiments)
correlation_dist = np.ndarray(experiments)
for k in range(experiments):
  r_s, c_c = compare_correlation_and_spearman_rank(n, 1.0)
  spearman_dist[k] = r_s
  correlation_dist[k] = c_c

print 'Spearman Rank Coefficient: ' + str(np.mean(spearman_dist))
# Compare to the regular correlation coefficient
print 'Correlation Coefficient: ' + str(np.mean(correlation_dist))


plt.hist(spearman_dist, bins=50, alpha=0.5)
plt.hist(correlation_dist, bins=50, alpha=0.5)
plt.legend(['Spearman Rank', 'Regular Correlation'])
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')

plt.show()

# Spearman correlation with a bit of noise to the data
n = 100
noises  = np.linspace(0,3,30)
experiments = 100
spearman = np.ndarray(len(noises))
correlation = np.ndarray(len(noises))

for k in range(len(noises)):
  # Run many experiments for each noise setting
  rank_coef = 0.0
  corr_coef = 0.0
  noise = noises[k]
  for m in range(experiments):
    r_s, c_c = compare_correlation_and_spearman_rank(n, noise)
    rank_coef += r_s
    corr_coef += c_c
  spearman[k] = rank_coef/experiments
  correlation[k] = corr_coef/experiments

plt.scatter(noises, spearman, color='r')
plt.scatter(noises, correlation)
plt.legend(['Spearman Rank', 'Regular Correlation'])
plt.xlabel('Amount of Noise')
plt.ylabel('Average Correlation Coefficient')
plt.show()


# Delay in correlation

n = 100

X = np.random.rand(n)
Xrank = stats.rankdata(X, method='average')
# n-2 is the second to last element
Yrank = stats.rankdata([1,1] + list(X[:(n-2)]), method='average')

diffs = Xrank - Yrank 
r_s = 1-6*sum(diffs**2)/(n*(n**2-1))
print r_s


# Built in Function
np.random.seed(161)
X = np.random.rand(10)
Y = np.random.rand(10)

r_s = stats.spearmanr(X,Y)
print 'Spearman Rank Coefficient: ', r_s[0]
print 'p-value: ', r_s[1]
