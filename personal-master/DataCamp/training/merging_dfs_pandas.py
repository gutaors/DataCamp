# Reading DataFrames from multiple files
# When data is spread among several files, you usually invoke pandas' read_csv() (or a similar data import function) multiple times to load the data into several DataFrames.
#
# The data files for this example have been derived from a list of Olympic medals awarded between 1896 & 2008 compiled by the Guardian.
#
# The column labels of each DataFrame are NOC, Country, & Total where NOC is a three-letter code for the name of the country and Total is the number of medals of that type won (bronze, silver, or gold).

# Import pandas as pd.
# Read the file 'Bronze.csv' into a DataFrame called bronze.
# Read the file 'Silver.csv' into a DataFrame called silver.
# Read the file 'Gold.csv' into a DataFrame called gold.
# Print the first 5 rows of the DataFrame gold. This has been done for you, so hit 'Submit Answer' to see the results.

# Import pandas
import pandas as pd

# Read 'Bronze.csv' into a DataFrame: bronze
bronze = pd.read_csv('Bronze.csv')

# Read 'Silver.csv' into a DataFrame: silver
silver = pd.read_csv('Silver.csv')

# Read 'Gold.csv' into a DataFrame: gold
gold = pd.read_csv('Gold.csv')

# Print the first five rows of gold
print(gold.head())

# Reading DataFrames from multiple files in a loop
# As you saw in the video, loading data from multiple files into DataFrames is more efficient in a loop or a list comprehension.
#
# Notice that this approach is not restricted to working with CSV files. That is, even if your data comes in other formats, as long as pandas has a suitable data import function, you can apply a loop or comprehension to generate a list of DataFrames imported from the source files.
#
# Here, you'll continue working with The Guardian's Olympic medal dataset.

# Reading multiple data files
filenames = ['a.csv', 'b.csv']
dataframes = [pd.read_csv(f) for f in filenames]

# Reading DataFrames from multiple files in a loop
# As you saw in the video, loading data from multiple files into DataFrames is more efficient in a loop or a list comprehension.
#
# Notice that this approach is not restricted to working with CSV files. That is, even if your data comes in other formats, as long as pandas has a suitable data import function, you can apply a loop or comprehension to generate a list of DataFrames imported from the source files.
#
# Here, you'll continue working with The Guardian's Olympic medal dataset.

# Create a list of file names called filenames with three strings 'Gold.csv', 'Silver.csv', & 'Bronze.csv'. This has been done for you.
# Use a for loop to create another list called dataframes containing the three DataFrames loaded from filenames:
# Iterate over filenames.
# Read each CSV file in filenames into a DataFrame and append it to dataframes by using pd.read_csv() inside a call to .append().
# Print the first 5 rows of the first DataFrame of the list dataframes. This has been done for you, so hit 'Submit Answer' to see the results.

# Import pandas
import pandas as pd

# Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']

# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))

# Print top 5 rows of 1st DataFrame in dataframes
print(dataframes[0].head())

# Combining DataFrames from multiple data files
# In this exercise, you'll combine the three DataFrames from earlier exercises - gold, silver, & bronze - into a single DataFrame called medals. The approach you'll use here is clumsy. Later on in the course, you'll see various powerful methods that are frequently used in practice for concatenating or merging DataFrames.
#
# Remember, the column labels of each DataFrame are NOC, Country, and Total, where NOC is a three-letter code for the name of the country and Total is the number of medals of that type won.

# Construct a copy of the DataFrame gold called medals using the .copy() method.
# Create a list called new_labels with entries 'NOC', 'Country', & 'Gold'. This is the same as the column labels from gold with the column label 'Total' replaced by 'Gold'.
# Rename the columns of medals by assigning new_labels to medals.columns.
# Create new columns 'Silver' and 'Bronze' in medals using silver['Total'] & bronze['Total'].
# Print the top 5 rows of the final DataFrame medals. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Make a copy of gold: medals
medals = gold.copy()

# Create list of new column labels: new_labels
new_labels = ['NOC', 'Country', 'Gold']

# Rename the columns of medals using new_labels
medals.columns = new_labels

# Add columns 'Silver' & 'Bronze' to medals
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']

# Print the head of medals
print(medals.head())

# Reindex from a df index
# Using .dropna removes entire rows in which null values occur

# Sorting DataFrame with the Index & columns
# It is often useful to rearrange the sequence of the rows of a DataFrame by sorting. You don't have to implement these yourself; the principal methods for doing this are .sort_index() and .sort_values().
#
# In this exercise, you'll use these methods with a DataFrame of temperature values indexed by month names. You'll sort the rows alphabetically using the Index and numerically using a column. Notice, for this data, the original ordering is probably most useful and intuitive: the purpose here is for you to understand what the sorting methods do.

# Read 'monthly_max_temp.csv' into a DataFrame called weather1 with 'Month' as the index.
# Sort the index of weather1 in alphabetical order using the .sort_index() method and store the result in weather2.
# Sort the index of weather1 in reverse alphabetical order by specifying the additional keyword argument ascending=False inside .sort_index().
# Use the .sort_values() method to sort weather1 in increasing numerical order according to the values of the column 'Max TemperatureF'

# Import pandas
import pandas as pd

# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col='Month')

# Print the head of weather1
print(weather1.head())

# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()

# Print the head of weather2
print(weather2.head())

# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending = False)

# Print the head of weather3
print(weather3.head())

# Sort weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values(['Max TemperatureF'])

# Print the head of weather4
print(weather4.head())

# Reindexing DataFrame from a list
# Sorting methods are not the only way to change DataFrame Indexes. There is also the .reindex() method.
#
# In this exercise, you'll reindex a DataFrame of quarterly-sampled mean temperature values to contain monthly samples (this is an example of upsampling or increasing the rate of samples, which you may recall from the pandas Foundations course).
#
# The original data has the first month's abbreviation of the quarter (three-month interval) on the Index, namely Apr, Jan, Jul, and Oct. This data has been loaded into a DataFrame called weather1 and has been printed in its entirety in the IPython Shell. Notice it has only four rows (corresponding to the first month of each quarter) and that the rows are not sorted chronologically.
#
# You'll initially use a list of all twelve month abbreviations and subsequently apply the .ffill() method to forward-fill the null entries when upsampling. This list of month abbreviations has been pre-loaded as year.

# Reorder the rows of weather1 using the .reindex() method with the list year as the argument, which contains the abbreviations for each month.
# Reorder the rows of weather1 just as you did above, this time chaining the .ffill() method to replace the null values with the last preceding non-null value.

# Import pandas
import pandas as pd

# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)

# Print weather2
print(weather2)

# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()

# Print weather3
print(weather3)

# Reindexing using another DataFrame Index
# Another common technique is to reindex a DataFrame using the Index of another DataFrame. The DataFrame .reindex() method can accept the Index of a DataFrame or Series as input. You can access the Index of a DataFrame with its .index attribute.
#
# The Baby Names Dataset from data.gov summarizes counts of names (with genders) from births registered in the US since 1881. In this exercise, you will start with two baby-names DataFrames names_1981 and names_1881 loaded for you.
#
# The DataFrames names_1981 and names_1881 both have a MultiIndex with levels name and gender giving unique labels to counts in each row. If you're interested in seeing how the MultiIndexes were set up, names_1981 and names_1881 were read in using the following commands:
#
# names_1981 = pd.read_csv('names1981.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# names_1881 = pd.read_csv('names1881.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# As you can see by looking at their shapes, which have been printed in the IPython Shell, the DataFrame corresponding to 1981 births is much larger, reflecting the greater diversity of names in 1981 as compared to 1881.
#
# Your job here is to use the DataFrame .reindex() and .dropna() methods to make a DataFrame common_names counting names from 1881 that were still popular in 1981.

# Create a new DataFrame common_names by reindexing names_1981 using the Index of the DataFrame names_1881 of older names.
# Print the shape of the new common_names DataFrame. This has been done for you. It should be the same as that of names_1881.
# Drop the rows of common_names that have null counts using the .dropna() method. These rows correspond to names that fell out of fashion between 1881 & 1981.
# Print the shape of the reassigned common_names DataFrame. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)

# parse_dates True to get datetime objects

# Scalar multiplication using asterisk
# weather.loc['2013-07-01':'2013-07-07', 'PrecipitationIn'] * 2.54

# df divide option with the option axis = 'rows'
week1_range.divide(week1_mean, axis='rows')

# Percentage changes using pct_change()

# Add dataframes: bronze + silver
bronze.add(silver)

# modify this behavior using fill_value = 0
# Chaining .add()

bronze.add(silver, fill_value=0).add(gold, fill_value = 0)

# Broadcasting in arithmetic formulas
# In this exercise, you'll work with weather data pulled from wunderground.com. The DataFrame weather has been pre-loaded along with pandas as pd. It has 365 rows (observed each day of the year 2013 in Pittsburgh, PA) and 22 columns reflecting different weather measurements each day.
#
# You'll subset a collection of columns related to temperature measurements in degrees Fahrenheit, convert them to degrees Celsius, and relabel the columns of the new DataFrame to reflect the change of units.
#
# Remember, ordinary arithmetic operators (like +, -, *, and /) broadcast scalar values to conforming DataFrames when combining scalars & DataFrames in arithmetic expressions. Broadcasting also works with pandas Series and NumPy arrays.

# Create a new DataFrame temps_f by extracting the columns 'Min TemperatureF', 'Mean TemperatureF', & 'Max TemperatureF' from weather as a new DataFrame temps_f. To do this, pass the relevant columns as a list to weather[].
# Create a new DataFrame temps_c from temps_f using the formula (temps_f - 32) * 5/9.
# Rename the columns of temps_c to replace 'F' with 'C' using the .str.replace('F', 'C') method on temps_c.columns.
# Print the first 5 rows of DataFrame temps_c. This has been done for you, so hit 'Submit Answer' to see the result!

# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')

# Print first 5 rows of temps_c
print(temps_c.head())

# Computing percentage growth of GDP
# Your job in this exercise is to compute the yearly percent-change of US GDP (Gross Domestic Product) since 2008.
#
# The data has been obtained from the Federal Reserve Bank of St. Louis and is available in the file GDP.csv, which contains quarterly data; you will resample it to annual sampling and then compute the annual growth of GDP. For a refresher on resampling, check out the relevant material from pandas Foundations.

# Read the file 'GDP.csv' into a DataFrame called gdp.
# Use parse_dates=True and index_col='DATE'.
# Create a DataFrame post2008 by slicing gdp such that it comprises all rows from 2008 onward.
# Print the last 8 rows of the slice post2008. This has been done for you. This data has quarterly frequency so the indices are separated by three-month intervals.
# Create the DataFrame yearly by resampling the slice post2008 by year. Remember, you need to chain .resample() (using the alias 'A' for annual frequency) with some kind of aggregation; you will use the aggregation method .last() to select the last element when resampling.
# Compute the percentage growth of the resampled DataFrame yearly with .pct_change() * 100

import pandas as pd

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv', index_col='DATE', parse_dates=True)

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

# Print yearly again
print(yearly)

# Converting currency of stocks
# In this exercise, stock prices in US Dollars for the S&P 500 in 2015 have been obtained from Yahoo Finance. The files sp500.csv for sp500 and exchange.csv for the exchange rates are both provided to you.
#
# Using the daily exchange rate to Pounds Sterling, your task is to convert both the Open and Close column prices.

# Read the DataFrames sp500 & exchange from the files 'sp500.csv' & 'exchange.csv' respectively..
# Use parse_dates=True and index_col='Date'.
# Extract the columns 'Open' & 'Close' from the DataFrame sp500 as a new DataFrame dollars and print the first 5 rows.
# Construct a new DataFrame pounds by converting US dollars to British pounds. You'll use the .multiply() method of dollars with exchange['GBP/USD'] and axis='rows'
# Print the first 5 rows of the new DataFrame pounds. This has been done for you, so hit 'Submit Answer' to see the results!.

# Import pandas
import pandas as pd

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', index_col='Date', parse_dates=True)

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', index_col='Date', parse_dates=True)

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis = 'rows')

# Print the head of pounds
print(pounds.head())

# append. Series and DataFrame method
s1.append(s2) # Stacks rows of s2 below s1

# concat is a pandas module function: accepts a list or seuqnce of several series of dfs to concatenate
pd.concat([s1, s2, s3]) # can stack row-wise or column-wise

# Using .reset_index()
northeast.append(south).reset_index(drop=True) # drops = True discards the old Index with repeated entries (rather than keeping it as a column in a DataFrame)

# use ignore_index
pd.concat([northeast, south], ignore_index=True)

# .append(): Series & DataFrame method
# Invocation:
# s1.append(s2)
# Stacks rows of s2 below s1
# Method for Series & DataFrames





# Appending Series with nonunique Indices
# The Series bronze and silver, which have been printed in the IPython Shell, represent the 5 countries that won the most bronze and silver Olympic medals respectively between 1896 & 2008. The Indexes of both Series are called Country and the values are the corresponding number of medals won.
#
# If you were to run the command combined = bronze.append(silver), how many rows would combined have? And how many rows would combined.loc['United States'] return? Find out for yourself by running these commands in the IPython Shell.

# Appending pandas Series
# In this exercise, you'll load sales data from the months January, February, and March into DataFrames. Then, you'll extract Series with the 'Units' column from each and append them together with method chaining using .append().
#
# To check that the stacking worked, you'll print slices from these Series, and finally, you'll add the result to figure out the total units sold in the first quarter.

# Read the files 'sales-jan-2015.csv', 'sales-feb-2015.csv' and 'sales-mar-2015.csv' into the DataFrames jan, feb, and mar respectively.
# Use parse_dates=True and index_col='Date'.
# Extract the 'Units' column of jan, feb, and mar to create the Series jan_units, feb_units, and mar_units respectively.
# Construct the Series quarter1 by appending feb_units to jan_units and then appending mar_units to the result. Use chained calls to the .append() method to do this.
# Verify that quarter1 has the individual Series stacked vertically. To do this:
# Print the slice containing rows from jan 27, 2015 to feb 2, 2015.
# Print the slice containing rows from feb 26, 2015 to mar 7, 2015.
# Compute and print the total number of units sold from the Series quarter1. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('sales-mar-2015.csv', index_col='Date', parse_dates=True)

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())

# Concatenating pandas Series along row axis
# Having learned how to append Series, you'll now learn how to achieve the same result by concatenating Series instead. You'll continue to work with the sales data you've seen previously. This time, the DataFrames jan, feb, and mar have been pre-loaded.
#
# Your job is to use pd.concat() with a list of Series to achieve the same result that you would get by chaining calls to .append().
#
# You may be wondering about the difference between pd.concat() and pandas' .append() method. One way to think of the difference is that .append() is a specific case of a concatenation, while pd.concat() gives you more flexibility, as you'll see in later exercises.

# Create an empty list called units. This has been done for you.
# Use a for loop to iterate over [jan, feb, mar]:
# In each iteration of the loop, append the 'Units' column of each DataFrame to units.
# Concatenate the Series contained in the list units into a longer Series called quarter1 using pd.concat().
# Specify the keyword argument axis='rows' to stack the Series vertically.
# Verify that quarter1 has the individual Series stacked vertically by printing slices. This has been done for you, so hit 'Submit Answer' to see the result!

# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])

# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis='rows')

# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Appending DataFrames with ignore_index
# In this exercise, you'll use the Baby Names Dataset (from data.gov) again. This time, both DataFrames names_1981 and names_1881 are loaded without specifying an Index column (so the default Indexes for both are RangeIndexes).
#
# You'll use the DataFrame .append() method to make a DataFrame combined_names. To distinguish rows from the original two DataFrames, you'll add a 'year' column to each with the year (1881 or 1981 in this case). In addition, you'll specify ignore_index=True so that the index values are not used along the concatenation axis. The resulting axis will instead be labeled 0, 1, ..., n-1, which is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information.

# Create a 'year' column in the DataFrames names_1881 and names_1981, with values of 1881 and 1981 respectively. Recall that assigning a scalar value to a DataFrame column broadcasts that value throughout.
# Create a new DataFrame called combined_names by appending the rows of names_1981 underneath the rows of names_1881. Specify the keyword argument ignore_index=True to make a new RangeIndex of unique integers for each row.
# Print the shapes of all three DataFrames. This has been done for you.
# Extract all rows from combined_names that have the name 'Morgan'. To do this, use the .loc[] accessor with an appropriate filter. The relevant column of combined_names here is 'name'.

# Add 'year' column to names_1881 and names_1981
names_1881['year'] = 1881
names_1981['year'] = 1981

# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981, ignore_index=True)

# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)

# Print all rows that contain the name 'Morgan'
print(combined_names.loc[combined_names['name']=='Morgan'])

# Concatenating pandas DataFrames along column axis
# The function pd.concat() can concatenate DataFrames horizontally as well as vertically (vertical is the default). To make the DataFrames stack horizontally, you have to specify the keyword argument axis=1 or axis='columns'.
#
# In this exercise, you'll use weather data with maximum and mean daily temperatures sampled at different rates (quarterly versus monthly). You'll concatenate the rows of both and see that, where rows are missing in the coarser DataFrame, null values are inserted in the concatenated DataFrame. This corresponds to an outer join (which you will explore in more detail in later exercises).
#
# The files 'quarterly_max_temp.csv' and 'monthly_mean_temp.csv' have been pre-loaded into the DataFrames weather_max and weather_mean respectively, and pandas has been imported as pd.

# Create weather_list, a list of the DataFrames weather_max and weather_mean.
# Create a new DataFrame called weather by concatenating weather_list horizontally.
# Pass the list to pd.concat() and specify the keyword argument axis=1 to stack them horizontally.
# Print the new DataFrame weather.

# Concatenate weather_max and weather_mean horizontally: weather
weather = pd.concat([weather_max, weather_mean], axis=1)

# Print weather
print(weather)

# Reading multiple files to build a DataFrame
# It is often convenient to build a large DataFrame by parsing many files as DataFrames and concatenating them all at once. You'll do this here with three files, but, in principle, this approach can be used to combine data from dozens or hundreds of files.
#
# Here, you'll work with DataFrames compiled from The Guardian's Olympic medal dataset.
#
# pandas has been imported as pd and two lists have been pre-loaded: An empty list called medals, and medal_types, which contains the strings 'bronze', 'silver', and 'gold'.

# Iterate over medal_types in the for loop.
# Inside the for loop:
# Create file_name using string interpolation with the loop variable medal. This has been done for you. The expression "%s_top5.csv" % medal evaluates as a string with the value of medal replacing %s in the format string.
# Create the list of column names called columns. This has been done for you.
# Read file_name into a DataFrame called medal_df. Specify the keyword arguments header=0, index_col='Country', and names=columns to get the correct row and column Indexes.
# Append medal_df to medals using the list .append() method.
# Concatenate the list of DataFrames medals horizontally (using axis='columns') to create a single DataFrame called medals. Print it in its entirety.

for medal in medal_types:
    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal

    # Create list of column names: columns
    columns = ['Country', medal]

    # Read file_name into a DataFrame: df
    medal_df = pd.read_csv(file_name, header=0, index_col='Country', names=columns)

    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals horizontally: medals
medals = pd.concat(medals, axis='columns')

# Print medals
print(medals)

# Concatenating vertically to get MultiIndexed rows
# When stacking a sequence of DataFrames vertically, it is sometimes desirable to construct a MultiIndex to indicate the DataFrame from which each row originated. This can be done by specifying the keys parameter in the call to pd.concat(), which generates a hierarchical index with the labels from keys as the outermost index label. So you don't have to rename the columns of each DataFrame as you load it. Instead, only the Index column needs to be specified.
#
# Here, you'll continue working with DataFrames compiled from The Guardian's Olympic medal dataset. Once again, pandas has been imported as pd and two lists have been pre-loaded: An empty list called medals, and medal_types, which contains the strings 'bronze', 'silver', and 'gold'.

# Within the for loop:
# Read file_name into a DataFrame called medal_df. Specify the index to be 'Country'.
# Append medal_df to medals.
# Concatenate the list of DataFrames medals into a single DataFrame called medals. Be sure to use the keyword argument keys=['bronze', 'silver', 'gold'] to create a vertically stacked DataFrame with a MultiIndex.
# Print the new DataFrame medals. This has been done for you, so hit 'Submit Answer' to see the result!

for medal in medal_types:
    file_name = "%s_top5.csv" % medal

    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, index_col='Country')

    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals: medals
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'])

# Print medals in entirety
print(medals)

# Slicing MultiIndexed DataFrames
# This exercise picks up where the last ended (again using The Guardian's Olympic medal dataset).
#
# You are provided with the MultiIndexed DataFrame as produced at the end of the preceding exercise. Your task is to sort the DataFrame and to use the pd.IndexSlice to extract specific slices. Check out this exercise from Manipulating DataFrames with pandas to refresh your memory on how to deal with MultiIndexed DataFrames.
#
# pandas has been imported for you as pd and the DataFrame medals is already in your namespace.

# Create a new DataFrame medals_sorted with the entries of medals sorted. Use .sort_index(level=0) to ensure the Index is sorted suitably.
# Print the number of bronze medals won by Germany and all of the silver medal data. This has been done for you.
# Create an alias for pd.IndexSlice called idx. A slicer pd.IndexSlice is required when slicing on the inner level of a MultiIndex.
# Slice all the data on medals won by the United Kingdom. To do this, use the .loc[] accessor with idx[:,'United Kingdom'], :.

# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)

# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:, 'United Kingdom'], :])

# Concatenating horizontally to get MultiIndexed columns
# It is also possible to construct a DataFrame with hierarchically indexed columns. For this exercise, you'll start with pandas imported and a list of three DataFrames called dataframes. All three DataFrames contain 'Company', 'Product', and 'Units' columns with a 'Date' column as the index pertaining to sales transactions during the month of February, 2015. The first DataFrame describes Hardware transactions, the second describes Software transactions, and the third, Service transactions.
#
# Your task is to concatenate the DataFrames horizontally and to create a MultiIndex on the columns. From there, you can summarize the resulting DataFrame and slice some information from it.

# Construct a new DataFrame february with MultiIndexed columns by concatenating the list dataframes.
# Use axis=1 to stack the DataFrames horizontally and the keyword argument keys=['Hardware', 'Software', 'Service'] to construct a hierarchical Index from each DataFrame.
# Print summary information from the new DataFrame february using the .info() method. This has been done for you.
# Create an alias called idx for pd.IndexSlice.
# Extract a slice called slice_2_8 from february (using .loc[] & idx) that comprises rows between Feb. 2, 2015 to Feb. 8, 2015 from columns under 'Company'.
# Print the slice_2_8. This has been done for you, so hit 'Submit Answer' to see the sliced data!

# Concatenate dataframes: february
february = pd.concat(dataframes, keys = ['Hardware', 'Software', 'Service'], axis = 1)

# Print february.info()
print(february.info())

# Assign pd.IndexSlice: idx
idx = pd.IndexSlice

# Create the slice: slice_2_8
slice_2_8 = february.loc['2015-02-02':'2015-02-08', idx[:, 'Company']]

# Print slice_2_8
print(slice_2_8)

# Excellent work! Working with MultiIndexes and MultiIndexed columns can seem tricky at first, but with practice, it will become second nature.

# Concatenating DataFrames from a dict
# You're now going to revisit the sales data you worked with earlier in the chapter. Three DataFrames jan, feb, and mar have been pre-loaded for you. Your task is to aggregate the sum of all sales over the 'Company' column into a single DataFrame. You'll do this by constructing a dictionary of these DataFrames and then concatenating them.

# Concatenating DataFrames from a dict
# You're now going to revisit the sales data you worked with earlier in the chapter. Three DataFrames jan, feb, and mar have been pre-loaded for you. Your task is to aggregate the sum of all sales over the 'Company' column into a single DataFrame. You'll do this by constructing a dictionary of these DataFrames and then concatenating them.

# Create a list called month_list consisting of the tuples ('january', jan), ('february', feb), and ('march', mar).
# Create an empty dictionary called month_dict.
# Inside the for loop:
# Group month_data by 'Company' and use .sum() to aggregate.
# Construct a new DataFrame called sales by concatenating the DataFrames stored in month_dict.
# Create an alias for pd.IndexSlice and print all sales by 'Mediacore'. This has been done for you, so hit 'Submit Answer' to see the result!

# Make the list of tuples: month_list
month_list = [('january', jan), ('february', feb), ('march', mar)]

# Create an empty dictionary: month_dict
month_dict = {}

for month_name, month_data in month_list:
    # Group month_data: month_dict[month_name]
    month_dict[month_name] = month_data.groupby('Company').sum()

# Concatenate data in month_dict: sales
sales = pd.concat(month_dict)

# Print sales
print(sales)

# Print all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])

# Using with arrays
import numpy as np
import pandas as pd

A = np.arange(8).reshape(2,4) + 0.1
B = np.arange(6).reshape(2,3) + 0.2
C = np.arange(12).reshape(3,4) + 0.3


# Both A, B must have the same nr of rows although the nr of columns can differ
# Stacking arrays horizontally
np.hstack([B, A])

# or

np.concatenate([B,A], axis=1) # axis 1 to get horizontal concatenation

# Stack a 2x4 matrix A and the 3x4 matrix C using np.vstack or np.concatenate with axis = 0
np.vstack([A,C])

np.concatenate([A,C], axis=0) #axis 0 is the default

# outer join: preserves indices in the original tables filling null values for missing rows
# inner join: intersection of index sets (only common labels)

# Concatenating DataFrames with inner join
# Here, you'll continue working with DataFrames compiled from The Guardian's Olympic medal dataset.
#
# The DataFrames bronze, silver, and gold have been pre-loaded for you.
#
# Your task is to compute an inner join.

# Construct a list of DataFrames called medal_list with entries bronze, silver, and gold.
# Concatenate medal_list horizontally with an inner join to create medals.
# Use the keyword argument keys=['bronze', 'silver', 'gold'] to yield suitable hierarchical indexing.
# Use axis=1 to get horizontal concatenation.
# Use join='inner' to keep only rows that share common index labels.
# Print the new DataFrame medals.

# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]

# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, keys = ['bronze', 'silver', 'gold'], axis = 1, join = 'inner')

# Print medals
print(medals)

# Resampling & concatenating DataFrames with inner join
# In this exercise, you'll compare the historical 10-year GDP (Gross Domestic Product) growth in the US and in China. The data for the US starts in 1947 and is recorded quarterly; by contrast, the data for China starts in 1961 and is recorded annually.
#
# You'll need to use a combination of resampling and an inner join to align the index labels. You'll need an appropriate offset alias for resampling, and the method .resample() must be chained with some kind of aggregation method (.pct_change() and .last() in this case).
#
# pandas has been imported as pd, and the DataFrames china and us have been pre-loaded, with the output of china.head() and us.head() printed in the IPython Shell.

# Make a new DataFrame china_annual by resampling the DataFrame china with .resample('A').last() (i.e., with annual frequency) and chaining two method calls:
# Chain .pct_change(10) as an aggregation method to compute the percentage change with an offset of ten years.
# Chain .dropna() to eliminate rows containing null values.
# Make a new DataFrame us_annual by resampling the DataFrame us exactly as you resampled china.
# Concatenate china_annual and us_annual to construct a DataFrame called gdp. Use join='inner' to perform an inner join and use axis=1 to concatenate horizontally.
# Print the result of resampling gdp every decade (i.e., using .resample('10A')) and aggregating with the method .last(). This has been done for you, so hit 'Submit Answer' to see the result!

# Resample and tidy china: china_annual
china_annual = china.resample('A').last().pct_change(10).dropna()

# Resample and tidy us: us_annual
us_annual = us.resample('A').last().pct_change(10).dropna()

# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual], join='inner', axis=1)

# Resample gdp and print
print(gdp.resample('10A').last())

# Stack dataframes vertically using append
# merge extends concat() with the ability to align rows using multiple columns

# pd.merge is by default an inner join

# Choose a particular column to merge on since it is difficult for all columns to match (which is the default behavior of merge)
pd.merge(bronze, gold, on='NOC') # column to merge on

# merging extends concatenation in allowing matches on multiple columns

# Use suffixes
pd.merge(bronze, gold, on=['NOC', 'Country'], suffixes=['_bronze', '_gold'])

# Specifying columns to merge
pd.merge(countries, cities, left_on='CITY NAME', right_on='City')

# Correct! Since the default strategy for pd.merge() is an inner join, combined will have 2 rows.

# Merging on a specific column
# This exercise follows on the last one with the DataFrames revenue and managers for your company. You expect your company to grow and, eventually, to operate in cities with the same name on different states. As such, you decide that every branch should have a numerical branch identifier. Thus, you add a branch_id column to both DataFrames. Moreover, new cities have been added to both the revenue and managers DataFrames as well. pandas has been imported as pd and both DataFrames are available in your namespace.
#
# At present, there should be a 1-to-1 relationship between the city and branch_id fields. In that case, the result of a merge on the city columns ought to give you the same output as a merge on the branch_id columns. Do they? Can you spot an ambiguity in one of the DataFrames?

# Using pd.merge(), merge the DataFrames revenue and managers on the 'city' column of each. Store the result as merge_by_city.
# Print the DataFrame merge_by_city. This has been done for you.
# Merge the DataFrames revenue and managers on the 'branch_id' column of each. Store the result as merge_by_id.
# Print the DataFrame merge_by_id. This has been done for you, so hit 'Submit Answer' to see the result!

# Merging on a specific column
# This exercise follows on the last one with the DataFrames revenue and managers for your company. You expect your company to grow and, eventually, to operate in cities with the same name on different states. As such, you decide that every branch should have a numerical branch identifier. Thus, you add a branch_id column to both DataFrames. Moreover, new cities have been added to both the revenue and managers DataFrames as well. pandas has been imported as pd and both DataFrames are available in your namespace.
#
# At present, there should be a 1-to-1 relationship between the city and branch_id fields. In that case, the result of a merge on the city columns ought to give you the same output as a merge on the branch_id columns. Do they? Can you spot an ambiguity in one of the DataFrames?

# Using pd.merge(), merge the DataFrames revenue and managers on the 'city' column of each. Store the result as merge_by_city.
# Print the DataFrame merge_by_city. This has been done for you.
# Merge the DataFrames revenue and managers on the 'branch_id' column of each. Store the result as merge_by_id.
# Print the DataFrame merge_by_id. This has been done for you, so hit 'Submit Answer' to see the result!

# Merge revenue with managers on 'city': merge_by_city
merge_by_city = pd.merge(revenue, managers, on='city')

# Print merge_by_city
print(merge_by_city)

# Merge revenue with managers on 'branch_id': merge_by_id
merge_by_id = pd.merge(revenue, managers, on='branch_id')

# Print merge_by_id
print(merge_by_id)

# Merging on columns with non-matching labels
# You continue working with the revenue & managers DataFrames from before. This time, someone has changed the field name 'city' to 'branch' in the managers table. Now, when you attempt to merge DataFrames, an exception is thrown:
#
# >>> pd.merge(revenue, managers, on='city')
# Traceback (most recent call last):
#     ... <text deleted> ...
#     pd.merge(revenue, managers, on='city')
#     ... <text deleted> ...
# KeyError: 'city'
# Given this, it will take a bit more work for you to join or merge on the city/branch name. You have to specify the left_on and right_on parameters in the call to pd.merge().
#
# As before, pandas has been pre-imported as pd and the revenue and managers DataFrames are in your namespace. They have been printed in the IPython Shell so you can examine the columns prior to merging.
#
# Are you able to merge better than in the last exercise? How should the rows with Springfield be handled?

# Merge the DataFrames revenue and managers into a single DataFrame called combined using the 'city' and 'branch' columns from the appropriate DataFrames.
# In your call to pd.merge(), you will have to specify the parameters left_on and right_on appropriately.
# Print the new DataFrame combined.

# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(revenue, managers, left_on='city', right_on='branch')

# Print combined
print(combined)

# Merging on multiple columns
# Another strategy to disambiguate cities with identical names is to add information on the states in which the cities are located. To this end, you add a column called state to both DataFrames from the preceding exercises. Again, pandas has been pre-imported as pd and the revenue and managers DataFrames are in your namespace.
#
# Your goal in this exercise is to use pd.merge() to merge DataFrames using multiple columns (using 'branch_id', 'city', and 'state' in this case).
#
# Are you able to match all your company's branches correctly?

# Create a column called 'state' in the DataFrame revenue, consisting of the list ['TX','CO','IL','CA'].
# Create a column called 'state' in the DataFrame managers, consisting of the list ['TX','CO','CA','MO'].
# Merge the DataFrames revenue and managers using three columns :'branch_id', 'city', and 'state'. Pass them in as a list to the on paramater of pd.merge().

# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX', 'CO', 'IL', 'CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX', 'CO', 'CA', 'MO']

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on = ['branch_id', 'city', 'state'])

# Print combined
print(combined)

# Default behavior for merge is how = 'inner'
# how='left' keeps all rows of the left df in the merged df

# The union of all rows from the left and right df can be preserved with an outer join. This is with how = 'outer'

# Which should we use?
# df1.append(df2): stacking vertically
# pd.concat([df1, df2]):
# stacking many horizontally or vertically
# simple inner or outer joins on Indexes

# pd.merge([df1, df2]): many joins on multiple columns

# Choosing a joining strategy
# Suppose you have two DataFrames: students (with columns 'StudentID', 'LastName', 'FirstName', and 'Major') and midterm_results (with columns 'StudentID', 'Q1', 'Q2', and 'Q3' for their scores on midterm questions).
#
# You want to combine the DataFrames into a single DataFrame grades, and be able to easily spot which students wrote the midterm and which didn't (their midterm question scores 'Q1', 'Q2', & 'Q3' should be filled with NaN values).
#
# You also want to drop rows from midterm_results in which the StudentID is not found in students.
#
# Which of the following strategies gives the desired result?

# Left & right merging on multiple columns
# You now have, in addition to the revenue and managers DataFrames from prior exercises, a DataFrame sales that summarizes units sold from specific branches (identified by city and state but not branch_id).
#
# Once again, the managers DataFrame uses the label branch in place of city as in the other two DataFrames. Your task here is to employ left and right merges to preserve data and identify where data is missing.
#
# By merging revenue and sales with a right merge, you can identify the missing revenue values. Here, you don't need to specify left_on or right_on because the columns to merge on have matching labels.
#
# By merging sales and managers with a left merge, you can identify the missing manager. Here, the columns to merge on have conflicting labels, so you must specify left_on and right_on. In both cases, you're looking to figure out how to connect the fields in rows containing Springfield.
#
# pandas has been imported as pd and the three DataFrames revenue, managers, and sales have been pre-loaded. They have been printed for you to explore in the IPython Shell.

# Execute a right merge using pd.merge() with revenue and sales to yield a new DataFrame revenue_and_sales.
# Use how='right' and on=['city', 'state'].
# Print the new DataFrame revenue_and_sales. This has been done for you.
# Execute a left merge with sales and managers to yield a new DataFrame sales_and_managers.
# Use how='left', left_on=['city', 'state'], and right_on=['branch', 'state'].
# Print the new DataFrame sales_and_managers. This has been done for you, so hit 'Submit Answer' to see the result!

# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue, sales, how='right', on = ['city', 'state'])

# Print revenue_and_sales
print(revenue_and_sales)

# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales, managers, how='left', left_on=['city', 'state'], right_on=['branch', 'state'])

# Print sales_and_managers
print(sales_and_managers)

# Merging DataFrames with outer join
# This exercise picks up where the previous one left off. The DataFrames revenue, managers, and sales are pre-loaded into your namespace (and, of course, pandas is imported as pd). Moreover, the merged DataFrames revenue_and_sales and sales_and_managers have been pre-computed exactly as you did in the previous exercise.
#
# The merged DataFrames contain enough information to construct a DataFrame with 5 rows with all known information correctly aligned and each branch listed only once. You will try to merge the merged DataFrames on all matching keys (which computes an inner join by default). You can compare the result to an outer join and also to an outer join with restricted subset of columns as keys.

# Merge sales_and_managers with revenue_and_sales. Store the result as merge_default.
# Print merge_default. This has been done for you.
# Merge sales_and_managers with revenue_and_sales using how='outer'. Store the result as merge_outer.
# Print merge_outer. This has been done for you.
# Merge sales_and_managers with revenue_and_sales only on ['city','state'] using an outer join. Store the result as merge_outer_on and hit 'Submit Answer' to see what the merged DataFrames look like!

# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers, revenue_and_sales)

# Print merge_default
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers, revenue_and_sales, how='outer')

# Print merge_outer
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers, revenue_and_sales, on = ['city', 'state'], how='outer')

# Print merge_outer_on
print(merge_outer_on)

# pd.merge_ordered() behaves like merge() when columns can be ordered. Default join for merge_ordered is an outer join contrasting the default inner join for merge

# Using merge_ordered()
# This exercise uses pre-loaded DataFrames austin and houston that contain weather data from the cities Austin and Houston respectively. They have been printed in the IPython Shell for you to examine.
#
# Weather conditions were recorded on separate days and you need to merge these two DataFrames together such that the dates are ordered. To do this, you'll use pd.merge_ordered(). After you're done, note the order of the rows before and after merging.
#

# Perform an ordered merge on austin and houston using pd.merge_ordered(). Store the result as tx_weather.
# Print tx_weather. You should notice that the rows are sorted by the date but it is not possible to tell which observation came from which city.
# Perform another ordered merge on austin and houston.
# This time, specify the keyword arguments on='date' and suffixes=['_aus','_hus'] so that the rows can be distinguished. Store the result as tx_weather_suff.
# Print tx_weather_suff to examine its contents. This has been done for you.
# Perform a third ordered merge on austin and houston.
# This time, in addition to the on and suffixes parameters, specify the keyword argument fill_method='ffill' to use forward-filling to replace NaN entries with the most recent non-null entry, and hit 'Submit Answer' to examine the contents of the merged DataFrames!

# Perform the first ordered merge: tx_weather
tx_weather = pd.merge_ordered(austin, houston)

# Print tx_weather
print(tx_weather)

# Perform the second ordered merge: tx_weather_suff
tx_weather_suff = pd.merge_ordered(austin, houston, on = 'date', suffixes=['_aus','_hus'])

# Print tx_weather_suff
print(tx_weather_suff)

# Perform the third ordered merge: tx_weather_ffill
tx_weather_ffill = pd.merge_ordered(austin, houston, on = 'date', suffixes=['_aus','_hus'], fill_method='ffill')

# Print tx_weather_ffill
print(tx_weather_ffill)

# Using merge_asof()
# Similar to pd.merge_ordered(), the pd.merge_asof() function will also merge values in order using the on column, but for each row in the left DataFrame, only rows from the right DataFrame whose 'on' column values are less than the left value will be kept.
#
# This function can be used to align disparate datetime frequencies without having to first resample.
#
# Here, you'll merge monthly oil prices (US dollars) into a full automobile fuel efficiency dataset. The oil and automobile DataFrames have been pre-loaded as oil and auto. The first 5 rows of each have been printed in the IPython Shell for you to explore.
#
# These datasets will align such that the first price of the year will be broadcast into the rows of the automobiles DataFrame. This is considered correct since by the start of any given year, most automobiles for that year will have already been manufactured.
#
# You'll then inspect the merged DataFrame, resample by year and compute the mean 'Price' and 'mpg'. You should be able to see a trend in these two columns, that you can confirm by computing the Pearson correlation between resampled 'Price' and 'mpg'.

# Merge auto and oil using pd.merge_asof() with left_on='yr' and right_on='Date'. Store the result as merged.
# Print the tail of merged. This has been done for you.
# Resample merged using 'A' (annual frequency), and on='Date'. Select [['mpg','Price']] and aggregate the mean. Store the result as yearly.
# Hit Submit Answer to examine the contents of yearly and yearly.corr(), which shows the Pearson correlation between the resampled 'Price' and 'mpg'.

# Merge auto and oil: merged
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')

# Print the tail of merged
print(merged.tail())

# Resample merged: yearly
yearly = merged.resample('A', on='Date')[['mpg', 'Price']].mean()

# Print yearly
print(yearly)

# print yearly.corr()
print(yearly.corr())

# Loading Olympic edition DataFrame
# In this chapter, you'll be using The Guardian's Olympic medal dataset.
#
# Your first task here is to prepare a DataFrame editions from a tab-separated values (TSV) file.
#
# Initially, editions has 26 rows (one for each Olympic edition, i.e., a year in which the Olympics was held) and 7 columns: 'Edition', 'Bronze', 'Gold', 'Silver', 'Grand Total', 'City', and 'Country'.
#
# For the analysis that follows, you won't need the overall medal counts, so you want to keep only the useful columns from editions: 'Edition', 'Grand Total', City, and Country

# Read file_path into a DataFrame called editions. The identifier file_path has been pre-defined with the filename 'Summer Olympic medallists 1896 to 2008 - EDITIONS.tsv'. You'll have to use the option sep='\t' because the file uses tabs to delimit fields (pd.read_csv() expects commas by default).
# Select only the columns 'Edition', 'Grand Total', 'City', and 'Country' from editions.
# Print the final DataFrame editions in entirety (there are only 26 rows). This has been done for you, so hit 'Submit Answer' to see the result!

# Loading IOC codes DataFrame
# Your task here is to prepare a DataFrame ioc_codes from a comma-separated values (CSV) file.
#
# Initially, ioc_codes has 200 rows (one for each country) and 3 columns: 'Country', 'NOC', & 'ISO code'.
#
# For the analysis that follows, you want to keep only the useful columns from ioc_codes: 'Country' and 'NOC' (the column 'NOC' contains three-letter codes representing each country).

# Read file_path into a DataFrame called ioc_codes. The identifier file_path has been pre-defined with the filename 'Summer Olympic medallists 1896 to 2008 - IOC COUNTRY CODES.csv'.
# Select only the columns 'Country' and 'NOC' from ioc_codes.
# Print the leading 5 and trailing 5 rows of the DataFrame ioc_codes (there are 200 rows in total). This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Create the file path: file_path
file_path = 'Summer Olympic medallists 1896 to 2008 - IOC COUNTRY CODES.csv'

# Load DataFrame from file_path: ioc_codes
ioc_codes = pd.read_csv(file_path)

# Extract the relevant columns: ioc_codes
ioc_codes = ioc_codes[['Country', 'NOC']]

# Print first and last 5 rows of ioc_codes
print(ioc_codes.head())
print(ioc_codes.tail())

# Building medals DataFrame
# Here, you'll start with the DataFrame editions from the previous exercise.
#
# You have a sequence of files summer_1896.csv, summer_1900.csv, ..., summer_2008.csv, one for each Olympic edition (year).
#
# You will build up a dictionary medals_dict with the Olympic editions (years) as keys and DataFrames as values.
#
# The dictionary is built up inside a loop over the year of each Olympic edition (from the Index of editions).
#
# Once the dictionary of DataFrames is built up, you will combine the DataFrames using pd.concat().

# Within the for loop:
# Create the file path. This has been done for you.
# Read file_path into a DataFrame. Assign the result to the year key of medals_dict.
# Select only the columns 'Athlete', 'NOC', and 'Medal' from medals_dict[year].
# Create a new column called 'Edition' in the DataFrame medals_dict[year] whose entries are all year.
# Concatenate the dictionary of DataFrames medals_dict into a DataFame called medals. Specify the keyword argument ignore_index=True to prevent repeated integer indices.
# Print the first and last 5 rows of medals. This has been done for you, so hit 'Submit Answer' to see the result!

# Import pandas
import pandas as pd

# Create empty dictionary: medals_dict
medals_dict = {}

for year in editions['Edition']:
    # Create the file path: file_path
    file_path = 'summer_{:d}.csv'.format(year)

    # Load file_path into a DataFrame: medals_dict[year]
    medals_dict[year] = pd.read_csv(file_path)

    # Extract relevant columns: medals_dict[year]
    medals_dict[year] = medals_dict[year][['Athlete', 'NOC', 'Medal']]

    # Assign year to column 'Edition' of medals_dict
    medals_dict[year]['Edition'] = year

# Concatenate medals_dict: medals
medals = pd.concat(medals_dict, ignore_index=True)

# Print first and last 5 rows of medals
print(medals.head())
print(medals.tail())

# Counting medals by country/edition in a pivot table
# Here, you'll start with the concatenated DataFrame medals from the previous exercise.
#
# You can construct a pivot table to see the number of medals each country won in each year. The result is a new DataFrame with the Olympic edition on the Index and with 138 country NOC codes as columns. If you want a refresher on pivot tables, it may be useful to refer back to the relevant exercises in Manipulating DataFrames with pandas.

# Construct a pivot table from the DataFrame medals, aggregating by count (by specifying the aggfunc parameter). Use 'Edition' as the index, 'Athlete' for the values, and 'NOC' for the columns.
# Print the first & last 5 rows of medal_counts. This has been done for you, so hit 'Submit Answer' to see the results!

# Construct the pivot_table: medal_counts
medal_counts = medals.pivot_table(index='Edition', values = 'Athlete', columns = 'NOC', aggfunc='count')

# Print the first & last 5 rows of medal_counts
print(medal_counts.head())
print(medal_counts.tail())

# Computing fraction of medals per Olympic edition
# In this exercise, you'll start with the DataFrames editions, medals, & medal_counts from prior exercises.
#
# You can extract a Series with the total number of medals awarded in each Olympic edition.
#
# The DataFrame medal_counts can be divided row-wise by the total number of medals awarded each edition; the method .divide() performs the broadcast as you require.
#
# This gives you a normalized indication of each country's performance in each edition.

# Set the index of the DataFrame editions to be 'Edition' (using the method .set_index()). Save the result as totals.
# Extract the 'Grand Total' column from totals and assign the result back to totals.
# Divide the DataFrame medal_counts by totals along each row. You will have to use the .divide() method with the option axis='rows'. Assign the result to fractions.
# Print first & last 5 rows of the DataFrame fractions. This has been done for you, so hit 'Submit Answer' to see the results!

# Set Index of editions: totals
totals = editions.set_index('Edition')

# Reassign totals['Grand Total']: totals
totals = totals['Grand Total']

# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals, axis = 'rows')

# Print first & last 5 rows of fractions
print(fractions.head())
print(fractions.tail())

# Computing percentage change in fraction of medals won
# Here, you'll start with the DataFrames editions, medals, medal_counts, & fractions from prior exercises.
#
# To see if there is a host country advantage, you first want to see how the fraction of medals won changes from edition to edition.
#
# The expanding mean provides a way to see this down each column. It is the value of the mean with all the data available up to that point in time. If you are interested in learning more about pandas' expanding transformations, this section of the pandas documentation has additional information.

# Create mean_fractions by chaining the methods .expanding().mean() to fractions.
# Compute the percentage change in mean_fractions down each column by applying .pct_change() and multiplying by 100. Assign the result to fractions_change.
# Reset the index of fractions_change using the .reset_index() method. This will make 'Edition' an ordinary column.
# Print the first and last 5 rows of the DataFrame fractions_change. This has been done for you, so hit 'Submit Answer' to see the results!

# Apply the expanding mean: mean_fractions
mean_fractions = fractions.expanding().mean()

# Compute the percentage change: fractions_change
fractions_change = mean_fractions.pct_change() * 100

# Reset the index of fractions_change: fractions_change
fractions_change = fractions_change.reset_index()

# Print first & last 5 rows of fractions_change
print(fractions_change.head())
print(fractions_change.tail())

# Building hosts DataFrame
# Your task here is to prepare a DataFrame hosts by left joining editions and ioc_codes.
#
# Once created, you will subset the Edition and NOC columns and set Edition as the Index.
#
# There are some missing NOC values; you will set those explicitly.
#
# Finally, you'll reset the Index & print the final DataFrame.

# Create the DataFrame hosts by doing a left join on DataFrames editions and ioc_codes (using pd.merge()).
# Clean up hosts by subsetting and setting the Index.
# Extract the columns 'Edition' and 'NOC'.
# Set 'Edition' column as the Index.
# Use the .loc[] accessor to find and assign the missing values to the 'NOC' column in hosts. This has been done for you.
# Reset the index of hosts using .reset_index(), which you'll need to save as the hosts DataFrame.
# Hit 'Submit Answer' to see what hosts looks like!

# Import pandas
import pandas as pd

# Left join editions and ioc_codes: hosts
hosts = pd.merge(editions, ioc_codes, how = 'left')

# Extract relevant columns and set index: hosts
hosts = hosts[['Edition', 'NOC']].set_index('Edition')

# Fix missing 'NOC' values of hosts
print(hosts.loc[hosts.NOC.isnull()])
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset Index of hosts: hosts
hosts = hosts.reset_index()

# Print hosts
print(hosts)

# Reshaping for analysis
# This exercise starts off with fractions_change and hosts already loaded.
#
# Your task here is to reshape the fractions_change DataFrame for later analysis.
#
# Initially, fractions_change is a wide DataFrame of 26 rows (one for each Olympic edition) and 139 columns (one for the edition and 138 for the competing countries).
#
# On reshaping with pd.melt(), as you will see, the result is a tall DataFrame with 3588 rows and 3 columns that summarizes the fractional change in the expanding mean of the percentage of medals won for each country in blocks.

# Create a DataFrame reshaped by reshaping the DataFrame fractions_change with pd.melt().
# You'll need to use the keyword argument id_vars='Edition' to set the identifier variable.
# You'll also need to use the keyword argument value_name='Change' to set the measured variables.
# Print the shape of the DataFrames reshaped and fractions_change. This has been done for you.
# Create a DataFrame chn by extracting all the rows from reshaped in which the three letter code for each country ('NOC') is 'CHN'.
# Print the last 5 rows of the DataFrame chn using the .tail() method. This has been done for you, so hit 'Submit Answer' to see the results!

# Import pandas
import pandas as pd

# Reshape fractions_change: reshaped
reshaped = pd.melt(fractions_change, id_vars='Edition', value_name='Change')

# Print reshaped.shape and fractions_change.shape
print(reshaped.shape, fractions_change.shape)

# Extract rows from reshaped where 'NOC' == 'CHN': chn
chn = reshaped[reshaped['NOC'] == 'CHN']

# Print last 5 rows of chn with .tail()
print(chn.tail())

# Merging to compute influence
# This exercise starts off with the DataFrames reshaped and hosts in the namespace.
#
# Your task is to merge the two DataFrames and tidy the result.
#
# The end result is a DataFrame summarizing the fractional change in the expanding mean of the percentage of medals won for the host country in each Olympic edition.

# Merge reshaped and hosts using an inner join. Remember, how='inner' is the default behavior for pd.merge().
# Print the first 5 rows of the DataFrame merged. This has been done for you. You should see that the rows are jumbled chronologically.
# Set the index of merged to be 'Edition' and sort the index.
# Print the first 5 rows of the DataFrame influence. This has been done for you, so hit 'Submit Answer' to see the results!

# Import pandas
import pandas as pd

# Merge reshaped and hosts: merged
merged = pd.merge(reshaped, hosts, how = 'inner')

# Print first 5 rows of merged
print(merged.head())

# Set Index of merged and sort it: influence
influence = merged.set_index('Edition').sort_index()

# Print first 5 rows of influence
print(influence.head())

# Plotting influence of host country
# This final exercise starts off with the DataFrames influence and editions in the namespace. Your job is to plot the influence of being a host country.

# Create a Series called change by extracting the 'Change' column from influence.
# Create a bar plot of change using the .plot() method with kind='bar'. Save the result as ax to permit further customization.
# Customize the bar plot of change to improve readability:
# Apply the method .set_ylabel("% Change of Host Country Medal Count") toax.
# Apply the method .set_title("Is there a Host Country Advantage?") to ax.
# Apply the method .set_xticklabels(editions['City']) to ax.
# Reveal the final plot using plt.show()

# Import pyplot
import matplotlib.pyplot as plt

# Extract influence['Change']: change
change = influence['Change']

# Make bar plot of change: ax
ax = change.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])

# Display the plot
plt.show()

