# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# ## Fish sleep and bacteria growth: A review of Statistical Thinking I and II

# ### EDA: Plot ECDFs of active bout length

bout_lengths_wt = [3.0000000000057985, 3.0000000000057985, 4.9799999999999045, 2.0399999999995093, 6.000000000001363, 3.0000000000057985, 0.9599999999977626, 10.020000000000095, 1.0199999999963438, 1.020000000003165, 0.9599999999977626, 1.980000000002633, 1.0199999999963438, 1.0199999999997544, 1.9800000000043383, 1.0199999999963438, 4.980000000003315, 1.0199999999946383, 0.9599999999977626, 4.020000000000437, 1.980000000006044, 2.9999999999972715, 3.9600000000035607, 2.9999999999989773, 4.020000000002142, 0.960000000004584, 3.9600000000035607, 4.980000000003315, 8.999999999998636, 1.0199999999963438, 1.0199999999997544, 4.9799999999999045, 0.9600000000028786, 9.000000000000343, 1.980000000000928, 10.019999999996683, 1.020000000003165, 0.960000000004584, 3.0000000000006817, 2.9999999999972715, 1.0199999999980491, 0.9600000000028786, 0.9600000000011732, 0.9600000000028786, 4.0200000000055525, 4.979999999998199, 1.0199999999963438, 2.040000000001214, 2.9999999999972715, 1.9799999999992224, 3.0000000000023874, 2.040000000001214, 1.0199999999997544, 0.9600000000028786, 1.9799999999992224, 3.0000000000023874, 0.960000000004584, 2.9999999999989773, 0.9600000000028786, 1.980000000002633, 2.9999999999972715, 4.019999999997026, 1.0199999999963438, 2.9999999999989773, 1.980000000002633, 7.9800000000005875, 2.9999999999989773, 1.0200000000014595, 2.9999999999972715, 1.020000000003165, 3.9600000000035607, 5.03999999999678, 1.020000000003165, 0.9599999999977626, 1.0199999999980491, 2.0399999999995093, 4.9799999999999045, 6.960000000000832, 1.0199999999997544, 4.020000000000437, 1.980000000000928, 2.9999999999989773, 3.0000000000006817, 1.9799999999992224, 1.0199999999997544, 1.0199999999997544, 1.0199999999997544, 1.0199999999997544, 4.020000000000437, 1.0199999999997544, 4.9799999999999045, 1.0199999999997544, 4.020000000000437, 4.020000000000437, 1.0199999999980491, 4.9799999999999045, 3.9600000000001496, 2.9999999999972715, 10.0200000000018, 1.9799999999975169]

bout_lengths_mut = [1.9799999999992224, 2.9999999999989773, 2.039999999994393, 4.980000000003315, 10.979999999997856, 13.019999999997367, 13.019999999999072, 16.980000000002633, 2.9999999999989773, 5.999999999999659, 15.0, 19.980000000003315, 6.0000000000030695, 7.979999999997176, 4.019999999995321, 10.020000000000095, 6.960000000000832, 2.9999999999989773, 6.0000000000030695, 2.9999999999989773, 1.0199999999963438, 5.9999999999979545, 5.039999999998486, 7.019999999997707, 17.99999999999727, 9.000000000000343, 1.0199999999963438, 2.040000000001214, 2.9999999999972715, 17.999999999998977, 6.960000000005947, 4.979999999998199, 1.980000000002633, 3.9600000000035607, 2.0399999999995093, 2.040000000001214, 5.03999999999678, 1.0199999999963438, 10.0200000000018, 1.0199999999963438, 4.9799999999999045, 4.019999999995321, 1.0199999999963438, 16.020000000003165, 1.0200000000014595, 0.9600000000011732, 9.000000000000343, 7.019999999997707, 1.980000000002633, 15.960000000001173, 2.9999999999989773, 20.03999999999849, 2.9999999999989773, 3.0000000000006817, 3.0000000000023874, 1.980000000000928, 4.9799999999999045, 3.0000000000023874, 0.9600000000011732, 5.999999999999659, 7.97999999999888, 1.9799999999992224, 3.0000000000023874, 1.9799999999992224, 1.980000000000928, 6.000000000001363, 0.9599999999994681, 5.999999999999659, 16.980000000000928, 36.00000000000137, 7.0199999999942975, 15.96000000000288, 8.039999999999168, 2.040000000001214, 1.9799999999975169, 19.9799999999982, 3.9600000000035607, 1.9800000000043383, 4.980000000003315, 12.960000000000491, 1.980000000002633, 17.999999999998977, 0.9600000000028786, 1.9799999999941065, 15.0, 32.0399999999961, 0.9599999999977626, 2.9999999999972715, 7.019999999997707, 1.020000000003165, 7.019999999997707, 19.02000000000044, 4.980000000003315, 7.0199999999994125, 3.960000000001856, 25.98000000000297, 4.980000000003315, 17.999999999998977, 2.039999999994393, 5.999999999999659]

# +
# Import the dc_stat_think module as dcst
import dc_stat_think as dcst

# Generate x and y values for plotting ECDFs
x_wt, y_wt = dcst.ecdf(bout_lengths_wt)
x_mut, y_mut = dcst.ecdf(bout_lengths_mut)

# Plot the ECDFs
_ = plt.plot(x_wt, y_wt, marker='.', linestyle='none')
_ = plt.plot(x_mut, y_mut, marker='.', linestyle='none')

# Make a legend, label axes, and show plot
_ = plt.legend(('wt', 'mut'))
_ = plt.xlabel('active bout length (min)')
_ = plt.ylabel('ECDF')
plt.show()

# -

# ### Parameter estimation: active bout length

# +
# Compute mean active bout length
mean_wt = np.mean(bout_lengths_wt)
mean_mut = np.mean(bout_lengths_mut)

# Draw bootstrap replicates
bs_reps_wt = dcst.draw_bs_reps(bout_lengths_wt, np.mean, size=10000)
bs_reps_mut = dcst.draw_bs_reps(bout_lengths_mut, np.mean, size=10000)

# Compute 95% confidence intervals
conf_int_wt = np.percentile(bs_reps_wt, [2.5, 97.5])
conf_int_mut = np.percentile(bs_reps_mut, [2.5, 97.5])

# Print the results
print("""
wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
""".format(mean_wt, *conf_int_wt, mean_mut, *conf_int_mut))
# -

# ### Permutation test: wild type versus heterozygote

bout_lengths_het = [11.999999999999318, 32.99999999999557, 0.9600000000028786, 4.9799999999999045, 1.9800000000043383, 1.9799999999975169, 4.9799999999999045, 30.0, 16.979999999999222, 22.019999999997708, 4.019999999995321, 30.0, 10.979999999997856, 0.9600000000028786, 1.0199999999963438, 11.03999999999985, 24.960000000001518, 1.0199999999963438, 5.999999999999659, 20.03999999999849, 49.019999999998724, 3.9600000000001496, 25.01999999999668, 5.040000000000191, 1.9799999999992224, 38.03999999999746, 5.999999999999659, 7.0199999999994125, 64.01999999999872, 1.9799999999992224, 1.0199999999997544, 1.9799999999992224, 4.019999999998731, 8.039999999999168, 4.019999999998731, 7.0199999999994125, 20.03999999999849, 1.0199999999997544, 0.9600000000011732, 7.9800000000057025, 0.9600000000028786, 0.9599999999977626, 1.0199999999997544, 7.97999999999888, 0.9600000000028786, 8.039999999999168, 0.960000000004584, 4.9799999999999045, 4.980000000003315, 1.980000000000928, 2.039999999994393, 1.020000000003165, 4.980000000003315, 0.9600000000028786, 2.9999999999989773, 0.9599999999977626, 1.0199999999946383, 10.980000000002976, 0.960000000004584, 1.0199999999946383, 3.0000000000006817, 0.9600000000028786, 0.9599999999977626, 4.979999999998199, 5.9999999999979545, 4.019999999998731, 3.960000000001856, 6.000000000001363, 2.9999999999972715, 2.039999999994393, 4.980000000003315, 1.9800000000043383, 2.0399999999960983, 1.980000000002633, 1.0199999999946383, 3.0000000000006817, 0.9600000000028786, 7.0199999999942975, 1.0200000000014595, 1.0199999999963438, 1.0199999999963438, 2.9999999999972715, 1.0199999999946383, 1.0200000000014595, 6.9599999999974225, 3.0000000000023874, 0.9599999999977626, 1.0199999999980491, 1.9800000000043383, 5.039999999998486, 4.020000000000437, 0.9600000000028786, 2.9999999999972715, 0.9600000000028786, 2.9999999999972715, 3.960000000001856, 10.019999999996683, 2.9999999999989773, 0.9600000000011732, 1.0199999999963438]

# +
# Compute the difference of means: diff_means_exp
diff_means_exp = np.mean(bout_lengths_het) - np.mean(bout_lengths_wt)

# Draw permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(bout_lengths_het, bout_lengths_wt, 
                               dcst.diff_of_means, size=10000)

# Compute the p-value: p_val
p_val = np.sum(perm_reps >= diff_means_exp) / len(perm_reps)

# Print the result
print('p =', p_val)
# -

# ### Bootstrap hypothesis test

# +
# Concatenate arrays: bout_lengths_concat
bout_lengths_concat = np.concatenate((bout_lengths_wt, bout_lengths_het))

# Compute mean of all bout_lengths: mean_bout_length
mean_bout_length = np.mean(bout_lengths_concat)

# Generate shifted arrays
wt_shifted = bout_lengths_wt - np.mean(bout_lengths_wt) + mean_bout_length
het_shifted = bout_lengths_het - np.mean(bout_lengths_het) + mean_bout_length

# Compute 10,000 bootstrap replicates from shifted arrays
bs_reps_wt = dcst.draw_bs_reps(wt_shifted, np.mean, size = 10000)
bs_reps_het = dcst.draw_bs_reps(het_shifted, np.mean, size = 10000)

# Get replicates of difference of means: bs_replicates
bs_reps = bs_reps_het - bs_reps_wt

# Compute and print p-value: p
p = np.sum(bs_reps >= diff_means_exp) / len(bs_reps)
print('p-value =', p)
# -

# ### Assessing the growth rate

df = pd.read_csv('park_bacterial_growth.csv', comment = '#')
df.head()

t = df['time (hr)'].values
bac_area = df['bacterial area (sq. microns)'].values

# +
# Compute logarithm of the bacterial area: log_bac_area
log_bac_area = np.log(bac_area)

# Compute the slope and intercept: growth_rate, log_a0
growth_rate, log_a0 = np.polyfit(t, log_bac_area, 1)

# Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
growth_rate_bs_reps, log_a0_bs_reps = \
            dcst.draw_bs_pairs_linreg(t, log_bac_area, size=10000)
    
# Compute confidence intervals: growth_rate_conf_int
growth_rate_conf_int = np.percentile(growth_rate_bs_reps, [2.5, 97.5])

# Print the result to the screen
print("""
Growth rate: {0:.4f} sq. µm/hour
95% conf int: [{1:.4f}, {2:.4f}] sq. µm/hour
""".format(growth_rate, *growth_rate_conf_int))
# -

# ### Plotting the growth curve

# +
# Plot data points in a semilog-y plot with axis labeles
_ = plt.semilogy(t, bac_area, marker='.', linestyle='none')

# Generate x-values for the bootstrap lines: t_bs
t_bs = np.array([0, 14])

# Plot the first 100 bootstrap lines
for i in range(100):
    y = np.exp(growth_rate_bs_reps[i] * t_bs + log_a0_bs_reps[i])
    _ = plt.semilogy(t_bs, y, linewidth=0.5, alpha=0.05, color='red')
    
# Label axes and show plot
_ = plt.xlabel('time (hr)')
_ = plt.ylabel('area (sq. µm)')
plt.show()
# -

# ## Analysis of results of the 2015 FINA World Swimming Championships

# ### Graphical EDA of men's 200 free heats

mens_200_free_heats = [118.32, 107.73, 107.0, 106.39, 108.75, 117.74, 108.43, 111.96, 114.36, 121.77, 108.23, 107.47, 118.41, 108.29, 106.0, 109.32, 111.49, 112.92, 117.38, 110.95, 108.27, 111.78, 107.87, 110.77, 109.05, 111.0, 108.77, 106.1, 106.61, 113.68, 108.2, 106.2, 111.01, 109.25, 112.0, 118.55, 109.56, 108.18, 111.67, 108.09, 110.04, 113.97, 109.91, 112.12, 111.65, 110.18, 116.36, 124.59, 115.59, 121.01, 106.88, 108.96, 109.09, 108.67, 109.6, 111.85, 118.54, 108.12, 124.38, 107.17, 107.48, 106.65, 106.91, 140.68, 117.93, 120.66, 111.29, 107.1, 108.49, 112.43, 110.61, 110.38, 109.87, 106.73, 107.18, 110.98, 108.55, 114.31, 112.05]

# +
# Generate x and y values for ECDF: x, y
x, y = dcst.ecdf(mens_200_free_heats)

# Plot the ECDF as dots
plt.plot(x, y, marker = '.', linestyle = 'none')

# Label axes and show plot
plt.xlabel('time (s)')
plt.ylabel('ECDF')
plt.show()

# -

# ### 200 m free time with confidence interval

# +
# Compute mean and median swim times
mean_time = np.mean(mens_200_free_heats)
median_time = np.median(mens_200_free_heats)

# Draw 10,000 bootstrap replicates of the mean and median
bs_reps_mean = dcst.draw_bs_reps(mens_200_free_heats, np.mean, size = 10000)
bs_reps_median = dcst.draw_bs_reps(mens_200_free_heats, np.median, size = 10000)


# Compute the 95% confidence intervals
conf_int_mean = np.percentile(bs_reps_mean, [2.5, 97.5])
conf_int_median = np.percentile(bs_reps_median, [2.5, 97.5])

# Print the result to the screen
print("""
mean time: {0:.2f} sec.
95% conf int of mean: [{1:.2f}, {2:.2f}] sec.

median time: {3:.2f} sec.
95% conf int of median: [{4:.2f}, {5:.2f}] sec.
""".format(mean_time, *conf_int_mean, median_time, *conf_int_median))
# -

# ### EDA: finals versus semifinals

semi_times = np.array([53.0, 24.32, 52.84, 24.22, 57.59, 116.95, 58.56, 27.7, 126.56, 59.05, 27.83, 127.57, 25.81, 24.38, 27.41, 58.05, 128.99, 24.52, 57.52, 142.82, 128.16, 31.03, 59.33, 27.18, 57.63, 66.28, 143.06, 57.36, 25.79, 116.44, 53.91, 127.08, 27.67, 127.69, 141.99, 57.04, 25.27, 58.84, 27.63, 25.88, 142.9, 25.71, 24.5, 59.71, 27.88, 57.77, 126.64, 129.16, 28.01, 116.51, 126.18, 127.05, 129.04, 67.11, 30.9, 116.23, 66.95, 66.21, 30.78, 126.36, 66.64, 142.15, 142.88, 65.64, 29.98, 116.91, 53.38, 53.78, 24.23, 25.9, 25.91, 116.56, 128.74, 65.6, 30.14, 59.55, 142.72, 55.74, 52.78, 25.06, 24.31, 66.76, 30.39, 30.64, 53.81, 24.47, 142.04, 116.76, 59.42, 116.37, 53.92, 127.79, 30.25, 127.52, 59.63, 127.57])

final_times = np.array([52.52, 24.12, 52.82, 24.36, 57.67, 116.41, 58.26, 27.66, 125.81, 58.75, 27.92, 126.78, 25.93, 24.44, 27.26, 58.22, 128.66, 24.39, 57.69, 143.61, 128.51, 30.74, 59.02, 27.11, 57.85, 66.55, 142.76, 57.48, 25.37, 116.27, 54.76, 126.51, 27.58, 130.2, 142.76, 57.05, 25.34, 58.86, 27.73, 25.78, 142.76, 25.85, 24.51, 59.78, 27.99, 57.94, 126.78, 128.49, 28.17, 116.19, 126.84, 127.76, 129.53, 67.1, 31.12, 115.32, 67.6, 66.42, 30.11, 125.56, 66.43, 141.15, 143.19, 66.36, 30.14, 116.79, 53.58, 53.17, 24.22, 25.64, 26.2, 116.16, 127.64, 65.66, 30.13, 59.66, 143.59, 55.64, 52.7, 24.96, 24.31, 67.17, 30.05, 31.14, 53.93, 24.57, 142.44, 115.16, 59.4, 115.49, 54.0, 126.34, 30.2, 126.95, 59.99, 126.4])

# +
# Compute fractional difference in time between finals and semis
f = (final_times - semi_times) / final_times

# Generate x and y values for the ECDF: x, y
x, y = dcst.ecdf(f)

# Make a plot of the ECDF
plt.plot(x, y, marker = '.', linestyle = 'none')

# Label axes and show plot
_ = plt.xlabel('f')
_ = plt.ylabel('ECDF')
plt.show()
# -

# ### Parameter estimates of difference between finals and semifinals

# +
# Mean fractional time difference: f_mean
f_mean = np.mean(f)

# Get bootstrap reps of mean: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size = 10000)

# Compute confidence intervals: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Report
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))


# -

# ### Generating permutation samples

def swap_random(a,b):
    """Randomly swap entries in two arrays."""
    # Indices to swap
    swap_inds = np.random.random(size = len(a)) < 0.5
    
    # Make copies of arrays a and b for output
    a_out = np.copy(a)
    b_out = np.copy(b)
    
    # Swap values
    a_out[swap_inds] = b[swap_inds]
    b_out[swap_inds] = a[swap_inds]

    return a_out, b_out


# ### Hypothesis test: Do women swim the same way in semis and finals?

# +
# Set up array of permutation replicates
perm_reps = np.empty(1000)

for i in range(1000):
    # Generate a permutation sample
    semi_perm, final_perm = swap_random(semi_times, final_times)
    
    # Compute f from the permutation sample
    f = (semi_perm - final_perm) / semi_perm
    
    # Compute and store permutation replicate
    perm_reps[i] = np.mean(f)

# Compute and print p-value
print('p =', np.sum(perm_reps >= f_mean) / 1000)
# -

# ### EDA: Plot all your data

split_number = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

splits = [np.array([35.04, 36.39, 35.92, 36.23, 36.67, 36.76, 36.48, 36.85, 36.92,
       36.68, 36.97, 36.98]), np.array([34.14, 34.22, 33.67, 33.88, 34.15, 33.91, 34.41, 33.92, 34.36,
       34.38, 34.6 , 34.45]), np.array([31.8 , 31.91, 31.95, 32.04, 31.95, 31.65, 31.57, 31.39, 31.61,
       31.43, 31.46, 31.47]), np.array([33.16, 32.9 , 32.68, 32.84, 33.55, 33.74, 33.71, 33.6 , 33.71,
       33.12, 33.14, 32.79]), np.array([32.97, 32.83, 32.99, 32.94, 33.19, 33.6 , 33.72, 33.74, 33.82,
       33.67, 33.86, 33.59])]

# +
# Plot the splits for each swimmer
for splitset in splits:
    _ = plt.plot(split_number, splitset, linewidth=1, color='lightgray')

# Compute the mean split times
mean_splits = np.mean(splits, axis = 0)

# Plot the mean split times
_ = plt.plot(split_number, mean_splits, marker = '.', linewidth = 3, markersize = 12)

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()
# -

# ### Linear regression of average split time

# +
# Perform regression
slowdown, split_3 = np.polyfit(split_number, mean_splits, 1)

# Compute pairs bootstrap
bs_reps, _ = dcst.draw_bs_pairs_linreg(split_number, mean_splits, size = 10000)

# Compute confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Plot the data with regressions line
_ = plt.plot(split_number, mean_splits, marker='.', linestyle='none')
_ = plt.plot(split_number, slowdown * split_number + split_3, '-')

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

# Print the slowdown per split
print("""
mean slowdown: {0:.3f} sec./split
95% conf int of mean slowdown: [{1:.3f}, {2:.3f}] sec./split""".format(
    slowdown, *conf_int))

# -

# ### Hypothesis test: are they slowing down?

# +
# Observed correlation
rho = dcst.pearson_r(split_number, mean_splits)

# Initialize permutation reps
perm_reps_rho = np.empty(10000)

# Make permutation reps
for i in range(10000):
    # Scramble the split number array
    scrambled_split_number = np.random.permutation(split_number)
    
    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = dcst.pearson_r(scrambled_split_number, mean_splits)
    
# Compute and print p-value
p_val = np.sum(perm_reps_rho>= rho) / 10000
print('p =', p_val)
# -

# ## The "Current Controversy" of the 2013 World Championships

# ### ECDF of improvement from low to high lanes

swimtime_low_lanes = np.array([24.66, 23.28, 27.2, 24.95, 32.34, 24.66, 26.17, 27.93, 23.35, 22.93, 21.93, 28.33, 25.14, 25.19, 26.11, 31.31, 27.44, 21.85, 27.48, 30.66, 21.74, 23.22, 27.93, 21.42, 24.79, 26.46])

swimtime_high_lanes = np.array([24.62, 22.9, 27.05, 24.76, 30.31, 24.54, 26.12, 27.71, 23.15, 23.11, 21.62, 28.02, 24.73, 24.95, 25.83, 30.61, 27.04, 21.67, 27.16, 30.23, 21.51, 22.97, 28.05, 21.65, 24.54, 26.06])

# +
# Compute the fractional improvement of being in high lane: f
f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes

# Make x and y values for ECDF: x, y
x, y = dcst.ecdf(f)

# Plot the ECDFs as dots
plt.plot(x, y, marker = '.', linestyle = 'none')

# Label the axes and show the plot
plt.xlabel('f')
plt.ylabel('ECDF')
plt.show()
# -

# ### Estimation of mean improvement

# +
# Compute the mean difference: f_mean
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size = 10000)

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Print the result
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))
# -

# ### Hypothesis test: Does lane assignment affect performance?

# +
# Shift f: f_shift
f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean: bs_reps
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=100000)

# Compute and report the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000
print('p =', p_val)
# -

# ### Did the 2015 event have this problem?

swimtime_low_lanes_15 = np.array([27.66, 24.69, 23.29, 23.05, 26.87, 31.03, 22.04, 24.51, 21.86, 25.64, 25.91, 24.77, 30.14, 27.23, 24.31, 30.2, 26.86])
swimtime_high_lanes_15 = np.array([27.7, 24.64, 23.21, 23.09, 26.87, 30.74, 21.88, 24.5, 21.86, 25.9, 26.2, 24.73, 30.13, 26.92, 24.31, 30.25, 26.76])

# +
# Compute f and its mean
f = (swimtime_low_lanes_15 - swimtime_high_lanes_15) / swimtime_low_lanes_15
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates
bs_reps = dcst.draw_bs_reps(f, np.mean, size = 10000)

# Compute 95% confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Shift f
f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size = 100000)

# Compute the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000

# Print the results
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]
p-value: {3:.5f}""".format(f_mean, *conf_int, p_val))
# -

# ### EDA: mean differences between odd and even splits

lanes = np.array([1, 2, 3, 4, 5, 6, 7, 8])

f_13 = np.array([-0.01562214370476285, -0.014638103564590154, -0.009776733919830173, -0.005257125646248888, 0.002041043769013733, 0.003810139186796837, 0.0075664033273217385, 0.01525869402175616])

f_15 = np.array([-0.005160181993383321, -0.003929522771324137, -0.000992838476615289, 0.0005995269293851621, -0.0024240044622061228, -0.004510986845095341, 0.0004746700793632159, 0.0008196226209307321])

# +
# Plot the the fractional difference for 2013 and 2015
_ = plt.plot(lanes, f_13, marker = '.', markersize = 12, linestyle = 'none')
_ = plt.plot(lanes, f_15, marker = '.', markersize = 12, linestyle = 'none')

# Add a legend
_ = plt.legend((2013, 2015))

# Label axes and show plot
_ = plt.xlabel('lane')
_ = plt.ylabel('frac. diff. (odd - even)')
plt.show()

# -

# ### How does the current effect depend on lane position?

# +
# Compute the slope and intercept of the frac diff/lane curve
slope, intercept = np.polyfit(lanes, f_13, 1)

# Compute bootstrap replicates
bs_reps_slope, bs_reps_int = dcst.draw_bs_pairs_linreg(lanes, f_13, size = 10000)

# Compute 95% confidence interval of slope
conf_int = np.percentile(bs_reps_slope, [2.5, 97.5])

# Print slope and confidence interval
print("""
slope: {0:.5f} per lane
95% conf int: [{1:.5f}, {2:.5f}] per lane""".format(slope, *conf_int))

# x-values for plotting regression lines
x = np.array([1, 8])

# Plot 100 bootstrap replicate lines
for i in range(100):
    _ = plt.plot(x, bs_reps_slope[i] * x + bs_reps_int[i], 
                 color='red', alpha=0.2, linewidth=0.5)

_ = plt.plot(lanes, f_13, marker = '.', markersize = 12, linestyle = 'none')
# Update the plot
plt.draw()
plt.show()
# -

# ### Hypothesis test: can this be by chance?

# +
# Compute observed correlation: rho
rho = dcst.pearson_r(lanes, f_13)

# Initialize permutation reps: perm_reps_rho
perm_reps_rho = np.empty(10000)

# Make permutation reps
for i in range(10000):
    # Scramble the lanes array: scrambled_lanes
    scrambled_lanes = np.random.permutation(lanes)
    
    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = dcst.pearson_r(scrambled_lanes, f_13)
    
# Compute and print p-value
p_val = np.sum(perm_reps_rho >= rho) / 10000
print('p =', p_val)
# -

# ## Statistical seismology and the Parkfield region
#

df = pd.read_csv('parkfield_earthquakes_1950-2017.csv', comment = '#')
df.head()

# ### Parkfield earthquake magnitudes

mags = df.mag

# +
# Make the plot
plt.plot(*dcst.ecdf(mags), marker = '.', linestyle = 'none')

# Label axes and show plot
plt.xlabel('magnitude')
plt.ylabel('ECDF')
plt.show()


# -

# ### Computing the b-value

def b_value(mags, mt, perc=[2.5, 97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]

    # Compute b-value: b
    b = (np.mean(m) - mt) * np.log(10)

    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = dcst.draw_bs_reps(m, np.mean, size=n_reps)

        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)

        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)
    
        return b, conf_int


# ### The b-value for Parkfield

# +
mt = 3
# Compute b-value and 95% confidence interval
b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps=10000)

# Generate samples to for theoretical ECDF
m_theor = np.random.exponential(b/np.log(10), size=100000) + mt

# Plot the theoretical CDF
_ = plt.plot(*dcst.ecdf(m_theor), marker = '.', linestyle = 'none')

# Plot the ECDF (slicing mags >= mt)
_ = plt.plot(*dcst.ecdf(mags[mags >= mt]), marker='.', linestyle='none')

# Pretty up and show the plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
_ = plt.xlim(2.8, 6.2)
plt.show()

# Report the results
print("""
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]""".format(b, *conf_int))
# -

# ### Interearthquake time estimates for Parkfield

time_gap = np.array([24.06570841889117, 20.076659822039698, 21.018480492813143, 12.24640657084189, 32.054757015742645, 38.25325119780972])

# +
# Compute the mean time gap: mean_time_gap
mean_time_gap = np.mean(time_gap)

# Standard deviation of the time gap: std_time_gap
std_time_gap = np.std(time_gap)

# Generate theoretical Exponential distribution of timings: time_gap_exp
time_gap_exp = np.random.exponential(mean_time_gap, size=10000)

# Generate theoretical Normal distribution of timings: time_gap_norm
time_gap_norm = np.random.normal(mean_time_gap, std_time_gap, size=10000)

# Plot theoretical CDFs
_ = plt.plot(*dcst.ecdf(time_gap_exp))
_ = plt.plot(*dcst.ecdf(time_gap_norm))

# Plot Parkfield ECDF
_ = plt.plot(*dcst.ecdf(time_gap, formal=True, min_x=-10, max_x=50))

# Add legend
_ = plt.legend(('Exp.', 'Norm.'), loc='upper left')

# Label axes, set limits and show plot
_ = plt.xlabel('time gap (years)')
_ = plt.ylabel('ECDF')
_ = plt.xlim(-10, 50)
plt.show()
# -

# ### When will the next big Parkfield quake be?

today = 2019.435149441914
last_quake = 2004.74

# +
# Draw samples from the Exponential distribution: exp_samples
exp_samples = np.random.exponential(mean_time_gap, size = 100000)

# Draw samples from the Normal distribution: norm_samples
norm_samples = np.random.normal(mean_time_gap, std_time_gap, size = 100000)

# No earthquake as of today, so only keep samples that are long enough
exp_samples = exp_samples[exp_samples > today - last_quake]
norm_samples = norm_samples[norm_samples > today - last_quake]

# Compute the confidence intervals with medians
conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + last_quake
conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + last_quake

# Print the results
print('Exponential:', conf_int_exp)
print('     Normal:', conf_int_norm)


# -

# ### Computing the K-S statistic

def ks_stat(data1, data2):
    # Compute ECDF from data: x, y
    x, y = dcst.ecdf(data1)
    
    # Compute corresponding values of the target CDF
    cdf = dcst.ecdf_formal(x, data2)

    # Compute distances between concave corners and CDF
    D_top = y -cdf

    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max((D_top, D_bottom))


# ### Drawing K-S replicates

def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):
    # Generate samples from target distribution
    x_f = f(*args, size = size)
    
    # Initialize K-S replicates
    reps = np.empty(n_reps)
    
    # Draw replicates
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args, size = n)
        
        # Compute K-S statistic
        reps[i] = dcst.ks_stat(x_samp, x_f)

    return reps


# ### The K-S test for Exponentiality

# +
# Draw target distribution: x_f
x_f = np.random.exponential(mean_time_gap, size = 10000)

# Compute K-S stat: d
d = dcst.ks_stat(time_gap, x_f)

# Draw K-S replicates: reps
reps = dcst.draw_ks_reps(len(time_gap), np.random.exponential, 
                         args=(mean_time_gap,), size=10000, n_reps=10000)

# Compute and print p-value
p_val = np.sum(reps >= d) / 10000
print('p =', p_val)
# -

# ## Earthquakes and oil mining in Oklahoma

df = pd.read_csv('oklahoma_earthquakes_1950-2017.csv', comment = '#')

time = pd.to_datetime(df.time)
mags = df.mag

# ### EDA: Plotting earthquakes over time

# +
# Plot time vs. magnitude
plt.plot(time, mags, marker = '.', linestyle = 'none', alpha = 0.1);

# Label axes and show the plot
plt.xlabel('time (year)');
plt.ylabel('magnitude');
# -

# ### Estimates of the mean interearthquake times

dt_pre = np.array([251.5, 295.7, 641.2, 1086.3, 317.6, 589.6, 483.6, 69.8, 692.6, 28.4, 469.8, 264.7, 76.6, 57.0, 105.9, 722.5, 233.8, 70.2, 115.0, 360.5, 836.3, 111.8, 141.8, 596.9, 167.6, 150.2, 327.4, 21.4, 184.3, 233.1, 378.9, 142.4, 89.7, 6.0, 19.0, 2.8, 11.4, 98.4, 12.0, 4.8, 20.4, 36.3, 0.7, 56.1, 22.3, 2.0, 0.7, 11.9, 8.7, 4.7, 0.9, 6.9])

dt_post = np.array([1.2, 0.0, 1.9, 6.7, 19.9, 1.9, 4.5, 6.7, 1.6, 5.9, 4.4, 1.1, 0.1, 0.5, 1.4, 8.7, 17.4, 3.6, 2.9, 0.5, 22.0, 38.4, 48.5, 29.6, 3.7, 11.2, 1.1, 3.0, 5.6, 6.4, 11.7, 12.3, 30.1, 0.0, 3.2, 0.0, 0.4, 13.5, 12.4, 3.6, 18.4, 55.0, 19.8, 122.9, 17.7, 13.1, 5.2, 6.6, 22.8, 22.8, 0.1, 7.9, 0.0, 0.1, 0.1, 0.1, 0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.2, 0.2, 0.1, 0.4, 0.0, 0.7, 0.7, 0.1, 0.7, 1.4, 0.2, 0.0, 1.3, 0.8, 2.5, 1.6, 3.6, 3.0, 1.0, 0.1, 1.6, 5.7, 6.5, 15.9, 5.8, 3.1, 14.1, 21.1, 22.2, 9.5, 21.0, 2.8, 3.4, 5.8, 1.8, 2.0, 5.2, 6.5, 2.6, 10.1, 0.3, 33.0, 0.7, 0.9, 18.8, 27.1, 0.9, 6.8, 10.7, 21.9, 21.7, 8.5, 7.8, 4.2, 6.7, 20.0, 24.9, 0.6, 3.9, 18.4, 12.2, 0.6, 42.0, 4.6, 2.3, 5.0, 5.1, 4.8, 0.0, 25.6, 0.0, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.6, 0.1, 0.0, 4.3, 1.8, 0.9, 0.2, 0.3, 2.6, 0.3, 22.0, 7.3, 11.0, 2.0, 1.6, 5.4, 0.2, 0.3, 0.2, 0.3, 1.3, 6.4, 1.9, 0.2, 3.1, 1.2, 0.0, 23.1, 3.4, 3.3, 1.0, 7.2, 0.7, 1.4, 35.9, 8.2, 0.1, 2.6, 5.6, 0.2, 0.0, 3.3, 3.2, 3.7, 8.3, 5.7, 7.3, 0.1, 0.1, 0.0, 1.4, 1.1, 0.2, 2.9, 0.0, 2.7, 0.5, 0.4, 0.3, 0.8, 2.1, 3.3, 4.3, 4.3, 0.4, 3.6, 1.4, 0.0, 3.7, 1.7, 0.1, 0.2, 0.3, 9.3, 5.1, 2.4, 0.3, 2.5, 0.7, 0.7, 4.2, 0.2, 1.5, 0.6, 0.0, 0.1, 4.2, 0.8, 16.3, 2.4, 0.5, 0.0, 1.1, 1.5, 0.7, 3.3, 1.3, 0.6, 2.6, 1.9, 0.2, 0.2, 3.5, 0.1, 1.1, 1.1, 0.0, 0.3, 0.1, 0.9, 0.0, 1.2, 1.9, 0.3, 1.2, 0.1, 0.1, 0.1, 0.6, 0.7, 2.1, 0.1, 0.0, 0.2, 0.4, 2.8, 0.4, 3.0, 2.5, 0.0, 0.8, 0.8, 0.9, 0.9, 2.6, 1.7, 1.5, 0.4, 0.2, 1.4, 0.0, 0.1, 0.4, 0.4, 0.4, 0.3, 0.4, 0.7, 0.0, 0.0, 0.0, 0.7, 0.0, 2.2, 0.6, 2.3, 0.9, 0.2, 0.3, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 1.1, 0.3, 0.3, 0.4, 0.3, 0.3, 1.4, 0.2, 0.0, 0.2, 0.0, 0.1, 0.5, 0.7, 0.1, 1.1, 0.0, 0.4, 0.1, 0.1, 0.4, 0.1, 1.4, 0.6, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.5, 0.9, 0.1, 0.4, 1.0, 0.1, 0.7, 2.9, 2.0, 1.4, 0.0, 0.0, 0.9, 0.7, 1.2, 1.1, 0.6, 0.7, 1.0, 1.1, 2.2, 0.0, 0.5, 0.1, 0.5, 0.4, 0.4, 0.2, 0.4, 0.6, 2.9, 1.5, 0.4, 0.4, 0.2, 0.9, 1.5, 0.5, 0.2, 1.9, 0.8, 0.3, 0.8, 0.3, 0.7, 0.6, 0.4, 0.5, 0.4, 0.4, 0.1, 0.9, 0.1, 0.1, 0.3, 1.4, 0.1, 3.3, 5.2, 0.3, 0.0, 0.5, 0.8, 0.6, 0.9, 0.2, 0.0, 7.3, 0.3, 4.3, 0.6, 0.2, 0.7, 0.0, 0.0, 1.2, 0.6, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.7, 0.1, 0.4, 0.1, 2.2, 0.1, 0.2, 1.0, 0.1, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.3, 1.4, 0.4, 0.4, 0.9, 0.0, 0.3, 2.0, 0.1, 0.2, 0.0, 0.4, 0.7, 1.2, 0.1, 0.5, 0.6, 0.0, 0.3, 3.9, 1.0, 0.2, 0.0, 0.9, 0.7, 0.4, 0.6, 0.1, 0.1, 2.6, 0.5, 4.5, 0.2, 2.0, 2.7, 0.7, 0.3, 0.2, 1.6, 0.0, 0.3, 0.3, 0.7, 0.6, 0.0, 0.5, 1.1, 0.3, 0.2, 0.9, 0.1, 0.9, 1.0, 1.0, 0.0, 1.1, 0.1, 0.0, 0.0, 1.7, 0.3, 1.6, 1.1, 0.9, 1.9, 0.8, 0.0, 0.4, 0.0, 0.4, 0.1, 0.7, 0.7, 0.9, 0.0, 0.0, 0.0, 2.3, 0.7, 0.0, 2.4, 0.8, 0.8, 0.2, 1.5, 0.3, 0.2, 1.7, 0.4, 0.1, 0.1, 0.6, 0.2, 0.0, 0.4, 0.1, 2.2, 0.5, 0.0, 1.8, 1.3, 0.6, 0.1, 0.1, 0.5, 0.0, 0.4, 0.2, 2.6, 0.7, 0.4, 0.1, 0.3, 0.3, 0.5, 0.4, 0.2, 0.3, 0.0, 0.4, 0.7, 0.5, 0.0, 0.0, 0.1, 0.7, 0.5, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 1.8, 0.5, 0.1, 0.1, 0.7, 2.3, 1.0, 0.0, 0.6, 1.5, 0.5, 0.1, 0.0, 0.0, 0.7, 0.8, 0.0, 0.1, 1.1, 1.6, 0.3, 0.2, 0.4, 2.1, 0.2, 0.3, 0.1, 0.7, 1.5, 0.3, 0.1, 0.0, 2.1, 0.3, 0.3, 0.1, 0.2, 0.4, 0.2, 0.0, 0.1, 0.5, 0.6, 0.1, 1.8, 0.2, 0.0, 0.1, 0.1, 0.0, 0.8, 0.3, 0.2, 0.0, 0.1, 0.6, 0.9, 0.2, 0.2, 0.3, 1.3, 0.3, 0.4, 2.2, 0.2, 0.1, 0.9, 0.7, 1.6, 0.7, 1.1, 0.2, 0.0, 0.2, 1.0, 0.8, 1.2, 0.1, 0.6, 0.3, 0.5, 0.3, 0.1, 2.8, 0.6, 0.0, 0.6, 0.0, 0.5, 0.2, 0.1, 0.3, 0.0, 1.5, 0.4, 0.9, 0.2, 0.5, 0.6, 0.1, 0.0, 0.0, 0.0, 0.2, 0.2, 0.9, 0.7, 0.1, 0.5, 0.9, 0.8, 0.2, 0.2, 0.0, 0.3, 0.0, 2.0, 0.1, 0.5, 0.3, 0.2, 0.0, 0.2, 0.4, 0.0, 1.1, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.1, 0.5, 0.3, 0.1, 0.4, 0.2, 0.1, 0.1, 0.2, 1.1, 0.2, 1.9, 0.1, 0.1, 0.1, 1.2, 0.8, 0.2, 0.9, 0.7, 0.1, 0.1, 0.3, 0.2, 0.2, 1.1, 1.6, 0.1, 0.7, 0.7, 0.1, 0.9, 0.1, 0.3, 0.0, 0.6, 0.5, 0.0, 0.6, 0.3, 0.6, 0.8, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.6, 0.1, 0.9, 0.0, 2.2, 0.6, 1.5, 0.2, 0.1, 0.3, 0.1, 0.0, 0.4, 0.1, 0.3, 1.4, 2.0, 0.1, 0.8, 0.0, 0.4, 0.3, 0.2, 0.7, 0.9, 0.3, 0.1, 0.7, 0.1, 0.1, 0.3, 0.0, 0.4, 0.7, 0.6, 0.4, 0.4, 0.3, 0.8, 0.2, 0.1, 1.1, 0.6, 0.1, 0.3, 0.0, 0.7, 0.1, 0.2, 0.9, 0.4, 0.7, 0.4, 0.0, 0.4, 0.0, 0.5, 0.0, 0.0, 0.1, 0.2, 0.9, 1.7, 0.3, 0.6, 0.0, 0.0, 0.1, 0.6, 0.6, 0.0, 0.8, 0.1, 0.3, 0.4, 0.3, 0.5, 0.2, 0.5, 0.3, 0.3, 0.0, 0.4, 0.2, 0.3, 0.1, 2.2, 1.7, 0.2, 0.6, 0.2, 0.5, 0.7, 0.8, 0.1, 0.1, 0.5, 0.0, 0.2, 0.0, 0.0, 0.5, 0.2, 0.2, 0.1, 1.2, 0.1, 0.1, 0.6, 3.0, 0.4, 0.6, 0.0, 0.9, 0.3, 0.1, 0.3, 0.1, 0.1, 0.3, 0.3, 0.7, 0.1, 0.0, 0.0, 1.1, 0.4, 0.2, 1.4, 1.0, 0.9, 0.2, 0.1, 0.2, 0.5, 2.0, 0.0, 0.0, 0.1, 0.4, 0.1, 0.3, 0.9, 0.1, 1.1, 0.1, 1.1, 0.0, 0.1, 0.0, 0.3, 1.0, 0.3, 0.8, 0.1, 0.2, 3.2, 0.3, 0.7, 0.6, 1.5, 1.0, 0.2, 0.4, 1.4, 0.0, 0.7, 0.3, 1.6, 0.4, 0.0, 0.0, 0.5, 1.5, 0.0, 0.1, 0.6, 0.1, 0.0, 0.2, 0.2, 0.0, 0.3, 0.5, 0.2, 0.7, 0.8, 0.8, 0.4, 0.4, 0.0, 1.3, 0.6, 1.3, 0.0, 0.1, 0.3, 0.1, 1.0, 0.8, 0.9, 0.8, 1.0, 0.0, 0.4, 0.0, 0.6, 0.5, 0.4, 0.4, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 1.2, 0.5, 0.3, 0.7, 0.0, 0.0, 0.7, 1.0, 0.0, 0.3, 0.6, 0.5, 0.0, 0.0, 1.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.9, 0.8, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.5, 0.2, 0.9, 0.3, 0.0, 0.5, 0.6, 0.2, 0.0, 0.1, 0.0, 0.1, 0.7, 0.0, 0.3, 0.2, 0.4, 0.6, 1.4, 0.2, 0.3, 0.2, 0.4, 0.6, 0.1, 1.6, 0.7, 0.9, 0.7, 0.0, 0.0, 0.0, 0.3, 1.5, 0.0, 1.2, 2.7, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 0.4, 1.0, 0.1, 0.0, 0.0, 0.5, 0.2, 0.5, 0.7, 0.1, 0.9, 0.4, 0.4, 0.1, 0.2, 1.5, 0.0, 1.4, 1.4, 0.0, 0.4, 0.4, 0.2, 1.4, 0.1, 0.0, 0.1, 0.1, 0.3, 0.1, 0.4, 0.4, 0.1, 0.5, 0.1, 0.3, 0.0, 1.1, 0.1, 0.1, 0.7, 1.8, 0.2, 1.3, 0.6, 0.5, 1.3, 0.1, 1.0, 0.1, 0.3, 0.0, 0.2, 0.4, 0.9, 0.8, 0.7, 0.4, 0.8, 0.2, 1.0, 2.1, 1.1, 0.0, 0.1, 0.1, 0.2, 0.0, 0.8, 0.2, 0.5, 3.3, 0.8, 0.1, 0.1, 0.1, 0.3, 0.0, 0.7, 0.3, 0.0, 0.8, 0.4, 0.3, 0.8, 0.2, 0.0, 0.9, 1.1, 0.1, 0.7, 0.3, 0.2, 0.8, 0.7, 0.6, 0.3, 0.5, 0.0, 0.0, 0.2, 0.7, 0.0, 0.0, 0.3, 0.0, 0.6, 0.4, 0.5, 0.0, 1.3, 0.1, 0.5, 1.2, 0.0, 1.1, 0.0, 0.0, 0.2, 0.4, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.5, 0.2, 0.0, 0.4, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.8, 1.5, 0.3, 0.7, 0.1, 0.0, 0.2, 0.0, 0.6, 0.1, 0.0, 0.4, 0.4, 0.1, 0.5, 0.0, 0.3, 0.2, 1.3, 0.4, 0.8, 0.2, 0.9, 0.1, 0.3, 0.2, 0.6, 0.5, 0.0, 0.4, 1.4, 1.2, 0.6, 0.6, 0.6, 1.1, 0.5, 0.1, 0.0, 0.3, 0.5, 1.0, 1.1, 0.3, 0.1, 0.9, 2.1, 2.6, 0.7, 0.0, 0.0, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 1.1, 0.0, 0.2, 0.8, 0.1, 0.0, 0.5, 0.0, 0.2, 0.1, 0.8, 0.1, 0.0, 0.1, 0.1, 0.8, 0.1, 0.2, 0.0, 0.0, 0.1, 0.2, 0.0, 0.3, 0.2, 2.5, 0.5, 1.3, 0.8, 1.3, 0.5, 0.4, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0, 0.1, 0.1, 0.3, 0.2, 1.0, 0.5, 0.3, 0.2, 1.0, 0.2, 0.4, 1.1, 0.5, 1.8, 0.2, 0.4, 0.0, 0.1, 0.1, 1.1, 0.3, 0.4, 0.5, 0.4, 1.1, 0.0, 0.7, 1.8, 0.1, 0.0, 0.3, 2.0, 0.8, 0.4, 0.0, 0.1, 1.0, 0.5, 0.4, 0.0, 0.4, 0.3, 0.1, 0.7, 1.9, 0.1, 1.0, 0.2, 0.1, 0.2, 0.5, 0.6, 0.8, 0.3, 0.8, 0.0, 5.6, 0.7, 0.3, 0.9, 0.7, 0.0, 1.6, 0.1, 1.4, 0.1, 0.2, 0.4, 0.4, 0.5, 0.1, 0.3, 0.0, 0.1, 0.1, 0.3, 0.1, 0.2, 0.8, 0.8, 0.0, 0.1, 0.0, 0.4, 0.2, 0.0, 0.0, 1.3, 0.4, 0.2, 0.7, 0.4, 0.1, 0.1, 0.7, 1.2, 0.7, 0.0, 0.2, 0.7, 1.6, 0.3, 0.0, 0.6, 1.0, 0.4, 0.4, 0.9, 0.1, 0.9, 0.0, 0.3, 2.3, 0.5, 0.0, 1.2, 1.0, 0.5, 0.2, 1.3, 0.5, 0.1, 0.1, 0.3, 0.3, 0.3, 0.0, 0.0, 0.1, 0.4, 0.1, 0.9, 0.0, 0.2, 0.1, 0.2, 0.8, 0.3, 0.8, 0.1, 0.3, 0.0, 0.2, 2.0, 0.7, 0.1, 1.0, 0.0, 0.4, 4.6, 0.6, 0.0, 0.3, 1.1, 1.1, 0.1, 0.2, 0.2, 0.6, 0.6, 0.2, 0.0, 1.2, 0.2, 0.1, 0.1, 0.4, 2.4, 0.0, 0.0, 0.9, 0.0, 0.1, 0.4, 0.5, 0.1, 2.5, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.5, 0.4, 0.2, 0.6, 0.3, 0.2, 0.0, 0.0, 0.2, 0.0, 0.2, 0.1, 0.0, 0.4, 0.2, 0.8, 0.1, 0.6, 0.8, 0.8, 0.7, 2.3, 0.4, 0.4, 0.6, 0.2, 0.1, 0.1, 0.1, 0.0, 0.7, 0.1, 0.1, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.6, 0.8, 1.4, 0.1, 0.0, 0.1, 0.0, 0.1, 0.9, 0.6, 0.2, 0.3, 0.0, 0.2, 0.0, 0.2, 0.0, 3.0, 1.0, 0.1, 0.0, 0.3, 0.1, 0.1, 0.7, 0.0, 0.5, 0.6, 1.2, 0.2, 0.4, 0.3, 0.1, 0.9, 0.1, 0.1, 0.6, 0.0, 1.2, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 1.0, 1.6, 0.2, 0.9, 0.2, 0.4, 0.8, 0.4, 0.3, 0.2, 0.1, 0.4, 0.9, 0.4, 0.4, 0.0, 0.1, 0.2, 0.4, 0.3, 1.6, 0.4, 1.6, 0.1, 0.2, 0.1, 0.2, 0.9, 0.0, 0.4, 0.0, 0.3, 0.6, 0.2, 0.4, 0.6, 0.0, 0.7, 0.5, 0.1, 0.2, 0.3, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 1.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.5, 0.0, 0.1, 0.0, 1.5, 0.2, 0.0, 0.2, 0.1, 0.1, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1, 0.2, 0.0, 0.2, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.5, 0.1, 0.2, 0.2, 0.0, 0.0, 1.5, 0.0, 0.5, 0.1, 0.4, 0.3, 0.4, 0.0, 0.1, 0.1, 0.2, 0.1, 0.6, 0.9, 0.1, 0.3, 0.7, 0.5, 0.2, 1.1, 0.0, 0.1, 0.2, 0.2, 1.0, 0.2, 0.3, 0.2, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.6, 1.1, 0.4, 0.1, 0.2, 0.3, 0.2, 2.6, 0.6, 0.1, 1.0, 0.1, 0.0, 0.1, 0.9, 0.1, 0.4, 1.0, 0.7, 0.5, 0.7, 0.7, 0.9, 0.6, 0.1, 0.1, 0.0, 0.7, 0.0, 0.5, 0.1, 0.8, 0.7, 1.3, 1.9, 0.2, 0.1, 1.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.3, 0.0, 0.5, 0.1, 0.1, 0.7, 0.1, 0.3, 0.8, 0.1, 0.3, 0.0, 0.0, 0.4, 0.0, 0.8, 0.7, 0.1, 0.1, 0.3, 0.7, 0.4, 0.1, 1.1, 0.3, 0.0, 0.4, 0.1, 0.0, 0.3, 0.0, 0.0, 0.2, 0.1, 0.8, 0.9, 0.1, 0.4, 0.5, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.3, 0.9, 0.3, 0.0, 1.5, 0.3, 0.0, 0.1, 0.6, 0.2, 0.0, 0.5, 0.2, 2.0, 1.1, 1.4, 0.5, 0.2, 0.1, 1.0, 0.2, 0.1, 0.2, 0.2, 0.0, 0.3, 0.2, 0.5, 1.1, 0.9, 0.4, 0.7, 0.3, 0.8, 0.7, 0.2, 0.0, 1.7, 1.0, 0.1, 0.3, 0.1, 0.5, 1.1, 0.0, 1.5, 1.0, 1.1, 1.6, 0.0, 0.0, 0.7, 1.2, 0.2, 0.5, 0.0, 0.7, 0.0, 1.2, 0.0, 0.0, 2.0, 0.0, 0.2, 0.0, 1.5, 0.1, 1.2, 1.2, 0.3, 0.1, 0.0, 0.0, 0.4, 0.1, 0.1, 0.3, 0.2, 0.0, 0.0, 0.4, 0.4, 0.5, 0.2, 0.8, 1.1, 0.9, 0.1, 1.2, 0.6, 1.1, 0.0, 1.8, 2.3, 0.9, 0.7, 0.2, 0.3, 1.4, 0.1, 0.6, 2.3, 0.0, 0.4, 0.1, 0.4, 0.3, 0.2, 0.2, 0.2, 1.1, 0.1, 0.5, 0.2, 0.0, 0.2, 0.6, 0.0, 0.2, 0.6, 1.0, 0.2, 0.5, 0.0, 0.2, 1.1, 0.2, 0.4, 0.4, 2.5, 0.9, 0.1, 0.4, 0.3, 1.1, 1.2, 0.6, 1.6, 0.8, 0.4, 0.3, 1.3, 2.8, 0.2, 0.2, 0.1, 1.4, 0.9, 0.5, 0.0, 2.2, 0.8, 1.0, 1.1, 0.9, 2.3, 0.8, 0.8, 0.7, 1.3, 0.1, 1.8, 0.6, 0.2, 0.0, 0.0, 0.9, 1.0, 0.6, 0.1, 0.4, 1.1, 1.4, 0.4, 0.7, 0.6, 0.2, 1.3, 0.0, 0.0, 0.0, 0.2, 0.4, 1.0, 1.5, 0.2, 0.2, 0.6, 0.7, 0.6, 0.0, 0.7, 0.7, 1.1, 0.4, 1.3, 0.2, 1.4, 1.3, 0.0, 0.7, 0.0, 0.5, 0.3, 1.3, 0.4, 0.9, 0.3, 1.2, 1.1, 0.3, 0.5, 3.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.3, 0.2, 0.0, 0.1, 1.9, 1.8, 0.1, 0.1, 0.4, 0.8, 1.5, 0.2, 0.5, 0.0, 0.2, 1.4, 0.2, 0.6, 2.2, 0.7, 0.4, 1.4, 0.6, 4.1, 2.7, 4.3, 0.3, 0.3, 1.5, 0.3, 2.2, 0.3, 0.0, 0.0, 0.4, 0.2, 0.2, 1.5, 0.2, 1.8, 0.4, 0.9, 0.0, 0.7, 0.5, 0.4, 0.5, 2.5, 4.3, 2.9, 2.4, 0.0, 0.4, 0.2, 0.4, 0.4, 1.1, 0.2, 1.6, 0.0, 0.4, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.4, 0.1, 0.2, 1.0, 0.9, 0.0, 0.0, 0.3, 0.8, 1.0, 0.0, 0.0, 0.1, 0.1, 0.6, 0.2, 0.0, 0.5, 1.8, 0.7, 0.2, 0.0, 0.5, 0.1, 1.0, 0.2, 1.8, 0.2, 0.3, 0.0, 1.5, 1.4, 0.6, 0.5, 2.1, 1.6, 1.7, 0.3, 0.3, 2.0, 0.5, 0.5, 1.9, 2.2, 0.2, 0.9, 0.2, 0.1, 0.0, 0.0, 0.3, 0.4, 2.0, 1.0, 0.3, 0.8, 0.2, 0.1, 2.2, 1.4, 1.1, 2.2, 0.4, 1.1, 0.2, 0.4, 0.0, 0.8, 1.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.6, 1.5, 2.2, 1.0, 1.4, 0.1, 0.5, 0.2, 0.0, 0.9, 0.4, 0.3, 0.0, 1.9, 0.0, 0.0, 0.2, 0.4, 2.3, 0.4, 0.7, 0.7, 0.3, 0.0, 0.2, 0.1, 0.2, 1.3, 0.2, 1.9, 0.4, 0.1, 0.2, 1.0, 1.3, 0.6, 0.0, 0.9, 3.6, 0.4, 2.9, 0.3, 1.6, 0.3, 0.9, 2.0, 7.5, 0.2, 0.4, 0.9, 1.1, 2.6, 0.4, 0.7, 0.7, 2.6, 2.2, 1.1, 0.2, 0.1, 1.0, 0.9, 0.2, 0.2, 1.5, 0.5, 1.3, 0.0, 0.2, 0.1, 4.0, 1.8, 0.0, 0.9, 1.4, 1.4, 0.0, 0.2, 2.3, 4.7, 0.4, 1.6, 3.3, 4.7, 2.5, 1.5, 1.0, 0.3, 0.7, 0.7, 1.7, 1.8, 0.3, 0.5, 1.1, 1.3, 1.5, 3.2, 4.6, 1.3, 3.9, 4.7, 1.0, 2.4, 0.9, 3.5, 0.6, 0.2, 0.7, 0.2, 0.3, 1.6, 0.8, 2.2, 0.0, 0.2, 1.1, 6.3, 0.4, 0.2, 0.6, 6.2, 1.6, 0.5, 0.8, 0.0, 0.0, 0.4, 2.3, 1.5, 3.7, 2.7, 0.0, 0.3, 0.5, 1.5, 0.1, 0.2, 1.5, 4.4, 0.0, 2.8, 0.1, 1.8, 0.4, 1.6, 0.4, 2.0, 0.0, 2.3, 0.4, 0.2, 0.3, 0.2, 0.7, 1.4, 0.0, 1.3, 5.6, 0.5, 2.2, 1.9, 3.0, 0.1, 0.1, 0.6, 0.0, 0.0, 0.2, 0.3, 1.2, 1.4, 2.6, 1.8, 0.8, 2.6, 1.5, 0.8, 2.6, 1.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.1, 0.9, 0.0, 0.7, 2.2, 2.4, 0.6, 0.4, 1.4, 0.0, 0.8, 5.0, 2.2, 2.3, 7.5, 2.2, 1.5, 0.5])

# +
# Compute mean interearthquake time
mean_dt_pre = np.mean(dt_pre)
mean_dt_post = np.mean(dt_post)

# Draw 10,000 bootstrap replicates of the mean
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size = 10000)
bs_reps_post = dcst.draw_bs_reps(dt_post, np.mean, size = 10000)

# Compute the confidence interval
conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])
# Print the results
print("""1980 through 2009
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))

print("""
2010 through mid-2017
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))

# -

# ### Hypothesis test: did earthquake frequency change?

# +
# Compute the observed test statistic
mean_dt_diff = mean_dt_pre - mean_dt_post

# Shift the post-2010 data to have the same mean as the pre-2010 data
dt_post_shift = dt_post - mean_dt_post + mean_dt_pre

# Compute 10,000 bootstrap replicates from arrays
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size = 10000)
bs_reps_post = dcst.draw_bs_reps(dt_post_shift, np.mean, size = 10000)

# Get replicates of difference of means
bs_reps = bs_reps_pre - bs_reps_post

# Compute and print the p-value
p_val =np.mean(bs_reps >= mean_dt_diff) / 10000
print('p =', p_val)
# -

# ### EDA: Comparing magnitudes before and after 2010

# +
# Get magnitudes before and after 2010
mags_pre = mags[time.dt.year < 2010]
mags_post = mags[time.dt.year >= 2010]

# Generate ECDFs
_ = plt.plot(*dcst.ecdf(mags_pre), marker = '.', linestyle = 'none')
_ = plt.plot(*dcst.ecdf(mags_post), marker = '.', linestyle = 'none')

# Label axes and show plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
plt.legend(('1980 though 2009', '2010 through mid-2017'), loc='upper left')
plt.show()

# -

# ### Quantification of the b-values

mt = 3
# Compute b-value and confidence interval for pre-2010
b_pre, conf_int_pre = b_value(mags_pre, mt, perc=[2.5, 97.5], n_reps=10000)
# Compute b-value and confidence interval for post-2010
b_post, conf_int_post = b_value(mags_post, mt, perc=[2.5, 97.5], n_reps=10000)
# Report the results
print("""
1980 through 2009
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]

2010 through mid-2017
b-value: {3:.2f}
95% conf int: [{4:.2f}, {5:.2f}]
""".format(b_pre, *conf_int_pre, b_post, *conf_int_post))

# ### Hypothesis test: are the b-values different?

# +
mt = 3
# Only magnitudes above completeness threshold
mags_pre = mags_pre[mags_pre >= mt]
mags_post = mags_post[mags_post >= mt]

# Observed difference in mean magnitudes: diff_obs
diff_obs = np.mean(mags_post) - np.mean(mags_pre)

# Generate permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(mags_post, mags_pre, dcst.diff_of_means, size = 10000)

# Compute and print p-value
p_val = np.sum(perm_reps < diff_obs) / 10000
print('p =', p_val)
# -


