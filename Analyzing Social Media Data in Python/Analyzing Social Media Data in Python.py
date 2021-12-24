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

# + active=""
# 1 - Basics of Analyzing Twitter Data
# -



# +
## Setting up tweepy authentication

from tweepy import OAuthHandler
from tweepy import API

# Consumer key authentication
auth = OAuthHandler(consumer_key, consumer_secret)

# Access key authentication
auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth)
# -



# +
## Collecting data on keywords

from tweepy import Stream

# Set up words to track
keywords_to_track = ["#rstats", "#python"]

# Instantiate the SListener object 
listen = SListener(api)

# Instantiate the Stream object
stream = Stream(auth, listen)

# Begin collecting data
stream.filter(track = keywords_to_track)
# -



# +
## Loading and accessing tweets

# Load JSON
import json

# Convert from JSON to Python object
tweet = json.loads(tweet_json)

# Print tweet text
print(tweet['text'])

# Print tweet id
print(tweet['id'])
# -



# +
## Accessing user data

# Print user handle
print(tweet['user']['screen_name'])

# Print user follower count
print(tweet['user']['followers_count'])

# Print user location
print(tweet['user']['location'])

# Print user description
print(tweet['user']['description'])
# -



# +
## Accessing retweet data

# Print the text of the tweet
print(rt['text'])
# RT @hannawallach: ICYMI: NIPS/ICML/ICLR are looking for a 
# full-time programmer to run the conferences' submission/review processes. More in…

# Print the text of tweet which has been retweeted
print(rt['retweeted_status']['text'])
# ICYMI: NIPS/ICML/ICLR are looking for a full-time programmer 
# to run the conferences' submission/review processes. M… https://t.co/aB9Y5tTyHT

# Print the user handle of the tweet
print(rt['user']['screen_name'])
# alexhanna

# Print the user handle of the tweet which has been retweeted
print(rt['retweeted_status']['user']['screen_name'])
# hannawallach

# -



# + active=""
# 2 - Processing Twitter text
# -



# +
## Tweet Items and Tweet Flattening

# Print the tweet text
print(quoted_tweet['text'])

# Print the quoted tweet text
print(quoted_tweet['quoted_status']['text'])

# Print the quoted tweet's extended (140+) text
print(quoted_tweet['quoted_status']['extended_tweet']['full_text'])

# Print the quoted user location
print(quoted_tweet['quoted_status']['user']['location'])

____________________________________________________________

# Store the user screen_name in 'user-screen_name'
quoted_tweet['user-screen_name'] = quoted_tweet['quoted_status']['user']['screen_name']

# Store the quoted_status text in 'quoted_status-text'
quoted_tweet['quoted_status-text'] = quoted_tweet['quoted_status']['text']

# Store the quoted tweet's extended (140+) text in 
# 'quoted_status-extended_tweet-full_text'
quoted_tweet['quoted_status-extended_tweet-full_text'] = quoted_tweet['quoted_status']['extended_tweet']['full_text']
# -



# +
## A tweet flattening function

def flatten_tweets(tweets_json):
    """ Flattens out tweet dictionaries so relevant JSON
        is in a top-level dictionary."""
    tweets_list = []
    
    # Iterate through each tweet
    for tweet in tweets_json:
        tweet_obj = json.loads(tweet)
    
        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
            
        tweets_list.append(tweet_obj)
    return tweets_list
# -



# +
## Loading tweets into a DataFrame

# Import pandas
import pandas as pd

# Flatten the tweets and store in `tweets`
tweets = flatten_tweets(data_science_json)

# Create a DataFrame from `tweets`
ds_tweets = pd.DataFrame(tweets)

# Print out the first 5 tweets from this dataset
print(ds_tweets['text'].values[0:5])

# + active=""
# <script.py> output:
#     ["RT @Dennboss: Hahahah Efteling maakt Maxi-Cosi's voor in de Python, duidelijk een perfect uitgewerkte 1 april grap en toch zijn er van die…"
#      'RT @PythonWeekly: Python Weekly - Issue 338 https://t.co/7gJSoLJj3V  #python #django #flask #slack #blockchain #bitcoin #twilio #opencv #ma…'
#      'RT @dataandme: ICYMI, still 💜ing this: "Where do things live in R? R for Excel Users" by @StephdeSilva https://t.co/WiIIxFZzRX #rstats http…'
#      'RT @dataandme: 🕴@jaredlander knows how to put on a show…\n"New York R Conference" is gonna be 🔥\nhttps://t.co/BpFTYm1Peh via @rstatsnyc (duh)…'
#      'RT @llanga: I heard it\'s Py Day today so I made a thing! Meet "Black", the uncompromising #python code formatter!\n\nhttps://t.co/COA0YMfUNr…']
# -



# +
## Finding keywords

# Flatten the tweets and store them
flat_tweets = flatten_tweets(data_science_json)

# Convert to DataFrame
ds_tweets = pd.DataFrame(flat_tweets)

# Find mentions of #python in 'text'
python = ds_tweets['text'].str.contains('#python', case=False)

# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / ds_tweets.shape[0] )
# -



# +
## Looking for text in all the wrong places

def check_word_in_tweet(word, data):
    """Checks if a word is in a Twitter dataset's text. 
    Checks text and extended tweet (140+ character tweets) for tweets,
    retweets and quoted tweets.
    Returns a logical pandas Series.
    """
    contains_column = data['text'].str.contains(word, case = False)
    contains_column |= data['extended_tweet-full_text'].str.contains(word, case = False)
    contains_column |= data['quoted_status-text'].str.contains(word, case = False)
    contains_column |= data['quoted_status-extended_tweet-full_text'].str.contains(word, case = False)
    contains_column |= data['retweeted_status-text'].str.contains(word, case = False)
    contains_column |= data['retweeted_status-extended_tweet-full_text'].str.contains(word, case = False)
    
    return contains_column
# -



# +
## Comparing #python to #rstats

# Find mentions of #python in all text fields
python = check_word_in_tweet("#python", ds_tweets)

# Find mentions of #rstats in all text fields
rstats = check_word_in_tweet("#rstats", ds_tweets)

# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / ds_tweets.shape[0])

# Print proportion of tweets mentioning #rstats
print("Proportion of #rstats tweets:", np.sum(rstats) / ds_tweets.shape[0])

# <script.py> output:
#     Proportion of #python tweets: 0.5733333333333334
#     Proportion of #rstats tweets: 0.4693333333333333
# -



# +
## Creating time series data frame

# Print created_at to see the original format of datetime in Twitter data
print(ds_tweets['created_at'].head())

# Convert the created_at column to np.datetime object
ds_tweets['created_at'] = pd.to_datetime(ds_tweets['created_at'])

# Print created_at to see new format
print(ds_tweets['created_at'].head())

# Set the index of ds_tweets to created_at
ds_tweets = ds_tweets.set_index('created_at')
# -



# +
## Generating mean frequency

# Create a python column
ds_tweets['python'] = check_word_in_tweet('#python', ds_tweets)

# Create an rstats column
ds_tweets['rstats'] = check_word_in_tweet('#rstats', ds_tweets)
# -



# +
## Plotting mean frequency

# Average of python column by day
mean_python = ds_tweets['python'].resample('1 d').mean()

# Average of rstats column by day
mean_rstats = ds_tweets['rstats'].resample('1 d').mean()

# Plot mean python by day(green)/mean rstats by day(blue)
plt.plot(mean_python.index.day, mean_python, color = 'green')
plt.plot(mean_rstats.index.day, mean_rstats, color = 'blue')

# Add labels and show
plt.xlabel('Day'); plt.ylabel('Frequency')
plt.title('Language mentions over time')
plt.legend(('#python', '#rstats'))
plt.show()
# -



# +
## Loading VADER

# Load SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

# Instantiate new SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Generate sentiment scores
sentiment_scores = ds_tweets['text'].apply(sid.polarity_scores)
# -



# +
## Calculating sentiment scores

# Print out the text of a positive tweet
print(ds_tweets[sentiment > .6]['text'].values[0])

# Print out the text of a negative tweet
print(ds_tweets[sentiment < -.6]['text'].values[0])

# Generate average sentiment scores for #python
sentiment_py = sentiment[check_word_in_tweet('#python', ds_tweets)].resample('1 d').mean()

# Generate average sentiment scores for #rstats
sentiment_r = sentiment[check_word_in_tweet('#rstats', ds_tweets)].resample('1 d').mean()
# -



# +
## Plotting sentiment scores

# Import matplotlib
import matplotlib.pyplot as plt

# Plot average #python sentiment per day
plt.plot(sentiment_py.index.day, sentiment_py, color = 'green')

# Plot average #rstats sentiment per day
plt.plot(sentiment_r.index.day, sentiment_r, color = 'blue')

plt.xlabel('Day')
plt.ylabel('Sentiment')
plt.title('Sentiment of data science languages')
plt.legend(('#python', '#rstats'))
plt.show()
# -



# + active=""
# 3 - Twitter Networks
# -



# +
## Creating retweet network

# Import networkx
import networkx as nx

# Create retweet network from edgelist
G_rt = nx.from_pandas_edgelist(
    sotu_retweets,
    source = 'user-screen_name',
    target = 'retweeted_status-user-screen_name',
    create_using = nx.DiGraph())
 
# Print the number of nodes
print('Nodes in RT network:', len(G_rt.nodes()))

# Print the number of edges
print('Edges in RT network:', len(G_rt.edges()))
# -



# +
## Creating reply network

# Import networkx
import networkx as nx

# Create reply network from edgelist
G_reply = nx.from_pandas_edgelist(
    sotu_replies,
    source = 'user-screen_name',
    target = 'in_reply_to_screen_name',
    create_using = nx.DiGraph())
    
# Print the number of nodes
print('Nodes in reply network:', len(G_reply.nodes()))

# Print the number of edges
print('Edges in reply network:', len(G_reply.edges()))
# -



# +
## Visualizing retweet network

# Create random layout positions
pos = nx.random_layout(G_rt)

# Create size list
sizes = [x[1] for x in G_rt.degree()]

# Draw the network
nx.draw_networkx(G_rt, pos, 
    with_labels = False, 
    node_size = sizes,
    width = 0.1, alpha = 0.7,
    arrowsize = 2, linewidths = 0)

# Turn axis off and show
plt.axis('off'); plt.show()
# -



# +
## In-degree centrality

# Generate in-degree centrality for retweets 
rt_centrality = nx.in_degree_centrality(G_rt)

# Generate in-degree centrality for replies 
reply_centrality = nx.in_degree_centrality(G_reply)

# Store centralities in DataFrame
rt = pd.DataFrame(list(rt_centrality.items()), columns = column_names)
reply = pd.DataFrame(list(reply_centrality.items()), columns = column_names)

# Print first five results in descending order of centrality
print(rt.sort_values('degree_centrality', ascending = False).head())

# Print first five results in descending order of centrality
print(reply.sort_values('degree_centrality', ascending = False).head())
# -



# +
## Betweenness Centrality

# Generate betweenness centrality for retweets 
rt_centrality = nx.betweenness_centrality(G_rt)

# Generate betweenness centrality for replies 
reply_centrality = nx.betweenness_centrality(G_reply)

# Store centralities in data frames
rt = pd.DataFrame(rt_centrality.items(), columns = column_names)
reply = pd.DataFrame(reply_centrality.items(), columns = column_names)

# Print first five results in descending order of centrality
print(rt.sort_values('betweenness_centrality', ascending = False).head())

# Print first five results in descending order of centrality
print(reply.sort_values('betweenness_centrality', ascending = False).head())
# -



# +
## Ratios

# Calculate in-degrees and store in DataFrame
degree_rt = pd.DataFrame(list(G_rt.in_degree()), columns = column_names)
degree_reply = pd.DataFrame(list(G_reply.in_degree()), columns = column_names)

# Merge the two DataFrames on screen name
ratio = degree_rt.merge(degree_reply, on = 'screen_name', suffixes = ('_rt', '_reply'))

# Calculate the ratio
ratio['ratio'] = ratio['degree_reply'] / ratio['degree_rt']

# Exclude any tweets with less than 5 retweets
ratio = ratio[ratio['degree_rt'] >= 5]

# Print out first five with highest ratio
print(ratio.sort_values('ratio', ascending = False).head())
# -



# + active=""
# 4 - Putting Twitter data on the map
# -



# +
## Accessing user-defined location

# Print out the location of a single tweet
print(tweet_json['user']['location'])

# Flatten and load the SOTU tweets into a dataframe
tweets_sotu = pd.DataFrame(flatten_tweets(tweets_sotu_json))

# Print out top five user-defined locations
print(tweets_sotu['user-location'].value_counts().head())
# -



# +
## Accessing bounding box

def getBoundingBox(place):
    """ Returns the bounding box coordinates."""
    return place['bounding_box']['coordinates']

# Apply the function which gets bounding box coordinates
bounding_boxes = tweets_sotu['place'].apply(getBoundingBox)

# Print out the first bounding box coordinates
print(bounding_boxes.values[0])

# <script.py> output:
#     [[[-94.043628, 28.855128], [-94.043628, 33.019544], [-88.758389, 33.019544], [-88.758389, 28.855128]]]
# -



# +
## Calculating the centroid

def calculateCentroid(place):
    """ Calculates the centroid from a bounding box."""
    # Obtain the coordinates from the bounding box.
    coordinates = place['bounding_box']['coordinates'][0]
        
    longs = np.unique( [x[0] for x in coordinates] )
    lats  = np.unique( [x[1] for x in coordinates] )

    if len(longs) == 1 and len(lats) == 1:
        # return a single coordinate
        return (longs[0], lats[0])
    elif len(longs) == 2 and len(lats) == 2:
        # If we have two longs and lats, we have a box.
        central_long = np.sum(longs) / 2
        central_lat  = np.sum(lats) / 2
    else:
        raise ValueError("Non-rectangular polygon not supported: %s" % 
            ",".join(map(lambda x: str(x), coordinates)) )

    return (central_long, central_lat)
    
# Calculate the centroids of place     
centroids = tweets_sotu['place'].apply(calculateCentroid)
# -



# +
## Creating Basemap map

# Import Basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Set up the US bounding box
us_boundingbox = [-125, 22, -64, 50] 

# Set up the Basemap object
m = Basemap(llcrnrlon = us_boundingbox[0],
            llcrnrlat = us_boundingbox[1],
            urcrnrlon = us_boundingbox[2],
            urcrnrlat = us_boundingbox[3],
            projection='merc')

# Draw continents in white,
# coastlines and countries in gray
m.fillcontinents(color='white')
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')

# Draw the states and show the plot
m.drawstates(color='gray')
plt.show()
# -



# +
## Plotting centroid coordinates

# Calculate the centroids for the dataset
# and isolate longitudue and latitudes
centroids = tweets_sotu['place'].apply(calculateCentroid)
lon = [x[0] for x in centroids]
lat = [x[1] for x in centroids]

# Draw continents, coastlines, countries, and states
m.fillcontinents(color='white', zorder = 0)
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# Draw the points and show the plot
m.scatter(lon, lat, latlon = True, alpha = 0.7)
plt.show()
# -



# +
## Coloring by sentiment

# Generate sentiment scores
sentiment_scores = tweets_sotu['text'].apply(sid.polarity_scores)

# Isolate the compound element
sentiment_scores = [x['compound'] for x in sentiment_scores]

# Draw the points
m.scatter(lon, lat, latlon = True, 
           c = sentiment_scores,
           cmap = 'coolwarm', alpha = 0.7)
           
# Show the plot
plt.show()
# -




