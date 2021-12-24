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

# ## Basics of Analyzing Twitter Data

# ### Setting up tweepy authentication

consumer_key = 'X'
consumer_secret = 'X'
access_token = 'X'
access_token_secret = 'X'

# +
from tweepy import OAuthHandler
from tweepy import API

# Consumer key authentication
auth = OAuthHandler(consumer_key, consumer_secret)

# Access key authentication
auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth)
# -

# ### Collecting data on keywords

# +
from tweepy.streaming import StreamListener
import json
import time
import sys

class SListener(StreamListener):
    def __init__(self, api = None, fprefix = 'streamer'):
        self.api = api or API()
        self.counter = 0
        self.fprefix = fprefix
        self.output  = open('%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')


    def on_data(self, data):
        if  'in_reply_to_status' in data:
            self.on_status(data)
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return


    def on_status(self, status):
        self.output.write(status)
        self.counter += 1
        if self.counter >= 20000:
            self.output.close()
            self.output  = open('%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0
        return


    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return


    def on_limit(self, track):
        print("WARNING: Limitation notice received, tweets missed: %d" % track)
        return


    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return 


    def on_timeout(self):
        print("Timeout, sleeping for 60 seconds...")
        time.sleep(60)
        return 

# +
from tweepy import Stream

# Set up words to track
keywords_to_track = ['#rstats', '#python']

# Instantiate the SListener object 
listen = SListener(api)

# Instantiate the Stream object
stream = Stream(auth, listen)

# Begin collecting data
stream.filter(track = keywords_to_track)
# -

# ### Loading and accessing tweets

# +
# Load JSON
import json

# Convert from JSON to Python object
# tweet = json.loads(tweet_json)
# -

tweet = {'text': "Writing out the script of my @DataCamp class and I can't help but mentally read it back to myself in @hugobowne's voice.", 'in_reply_to_status_id': None, 'geo': None, 'in_reply_to_status_id_str': None, 'is_quote_status': False, 'in_reply_to_screen_name': None, 'id': 986973961295720449, 'in_reply_to_user_id_str': None, 'metadata': {'result_type': 'recent', 'iso_language_code': 'en'}, 'in_reply_to_user_id': None, 'lang': 'en', 'created_at': 'Thu Apr 19 14:25:04 +0000 2018', 'contributors': None, 'coordinates': None, 'favorited': False, 'retweet_count': 0, 'favorite_count': 1, 'user': {'statuses_count': 71840, 'follow_request_sent': False, 'time_zone': 'Eastern Time (US & Canada)', 'profile_use_background_image': False, 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme16/bg.gif', 'screen_name': 'alexhanna', 'translator_type': 'regular', 'lang': 'en', 'followers_count': 4267, 'verified': False, 'profile_sidebar_border_color': '666666', 'profile_text_color': '333333', 'id': 661613, 'following': False, 'is_translation_enabled': False, 'profile_sidebar_fill_color': 'CCCCCC', 'geo_enabled': True, 'created_at': 'Thu Jan 18 20:37:52 +0000 2007', 'notifications': False, 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/661613/1514976085', 'protected': False, 'listed_count': 246, 'profile_background_color': '000000', 'contributors_enabled': False, 'url': 'https://t.co/WGddk8Cc6v', 'is_translator': False, 'favourites_count': 23387, 'location': 'Toronto, ON', 'friends_count': 2801, 'profile_image_url': 'http://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'has_extended_profile': False, 'profile_background_tile': False, 'profile_link_color': '0671B8', 'description': 'Assistant professor @UofT. Protest, media, computation. Trans. Roller derby athlete @TOROLLERDERBY (Kate Silver #538). She/her.', 'entities': {'url': {'urls': [{'display_url': 'alex-hanna.com', 'url': 'https://t.co/WGddk8Cc6v', 'expanded_url': 'http://alex-hanna.com', 'indices': [0, 23]}]}, 'description': {'urls': []}}, 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'default_profile': False, 'name': 'Alex Hanna, Data Witch', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme16/bg.gif', 'utc_offset': -14400, 'default_profile_image': False, 'id_str': '661613'}, 'entities': {'urls': [], 'hashtags': [], 'user_mentions': [{'name': 'DataCamp', 'screen_name': 'DataCamp', 'indices': [29, 38], 'id': 1568606814, 'id_str': '1568606814'}, {'name': 'Hugo Bowne-Anderson', 'screen_name': 'hugobowne', 'indices': [101, 111], 'id': 1092509048, 'id_str': '1092509048'}], 'symbols': []}, 'place': None, 'truncated': False, 'retweeted': False, 'id_str': '986973961295720449'}

# +
# Print tweet text
print(tweet['text'])

# Print tweet id
print(tweet['id'])
# -

# ### Accessing user data

# +
# Print user handle
print(tweet['user']['screen_name'])

# Print user follower count
print(tweet['user']['followers_count'])

# Print user location
print(tweet['user']['location'])

# Print user description
print(tweet['user']['description'])
# -

# ### Accessing retweet data

rt = {'text': "RT @hannawallach: ICYMI: NIPS/ICML/ICLR are looking for a full-time programmer to run the conferences' submission/review processes. More in…", 'in_reply_to_status_id': None, 'geo': None, 'in_reply_to_status_id_str': None, 'is_quote_status': False, 'in_reply_to_screen_name': None, 'id': 986949027123154944, 'in_reply_to_user_id_str': None, 'metadata': {'result_type': 'recent', 'iso_language_code': 'en'}, 'in_reply_to_user_id': None, 'lang': 'en', 'created_at': 'Thu Apr 19 12:45:59 +0000 2018', 'contributors': None, 'coordinates': None, 'favorited': False, 'retweet_count': 37, 'favorite_count': 0, 'user': {'statuses_count': 71840, 'follow_request_sent': False, 'time_zone': 'Eastern Time (US & Canada)', 'profile_use_background_image': False, 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme16/bg.gif', 'screen_name': 'alexhanna', 'translator_type': 'regular', 'lang': 'en', 'followers_count': 4267, 'verified': False, 'profile_sidebar_border_color': '666666', 'profile_text_color': '333333', 'id': 661613, 'following': False, 'is_translation_enabled': False, 'profile_sidebar_fill_color': 'CCCCCC', 'geo_enabled': True, 'created_at': 'Thu Jan 18 20:37:52 +0000 2007', 'notifications': False, 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/661613/1514976085', 'protected': False, 'listed_count': 246, 'profile_background_color': '000000', 'contributors_enabled': False, 'url': 'https://t.co/WGddk8Cc6v', 'is_translator': False, 'favourites_count': 23387, 'location': 'Toronto, ON', 'friends_count': 2801, 'profile_image_url': 'http://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'has_extended_profile': False, 'profile_background_tile': False, 'profile_link_color': '0671B8', 'description': 'Assistant professor @UofT. Protest, media, computation. Trans. Roller derby athlete @TOROLLERDERBY (Kate Silver #538). She/her.', 'entities': {'url': {'urls': [{'display_url': 'alex-hanna.com', 'url': 'https://t.co/WGddk8Cc6v', 'expanded_url': 'http://alex-hanna.com', 'indices': [0, 23]}]}, 'description': {'urls': []}}, 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'default_profile': False, 'name': 'Alex Hanna, Data Witch', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme16/bg.gif', 'utc_offset': -14400, 'default_profile_image': False, 'id_str': '661613'}, 'retweeted_status': {'text': "ICYMI: NIPS/ICML/ICLR are looking for a full-time programmer to run the conferences' submission/review processes. M… https://t.co/aB9Y5tTyHT", 'in_reply_to_status_id': None, 'geo': None, 'in_reply_to_status_id_str': None, 'is_quote_status': False, 'in_reply_to_screen_name': None, 'id': 971171213216239616, 'in_reply_to_user_id_str': None, 'metadata': {'result_type': 'recent', 'iso_language_code': 'en'}, 'in_reply_to_user_id': None, 'possibly_sensitive': False, 'lang': 'en', 'created_at': 'Tue Mar 06 23:50:35 +0000 2018', 'contributors': None, 'coordinates': None, 'favorited': False, 'retweet_count': 37, 'favorite_count': 52, 'user': {'statuses_count': 1505, 'follow_request_sent': False, 'time_zone': 'Eastern Time (US & Canada)', 'profile_use_background_image': False, 'profile_background_image_url': 'http://pbs.twimg.com/profile_background_images/521040468528754688/_Ayh3ZCE.jpeg', 'screen_name': 'hannawallach', 'translator_type': 'none', 'lang': 'en', 'followers_count': 10614, 'verified': False, 'profile_sidebar_border_color': 'FFFFFF', 'profile_text_color': '333333', 'id': 823957466, 'following': True, 'is_translation_enabled': False, 'profile_sidebar_fill_color': 'DDEEF6', 'geo_enabled': False, 'created_at': 'Fri Sep 14 20:38:24 +0000 2012', 'notifications': False, 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/823957466/1347986011', 'protected': False, 'listed_count': 499, 'profile_background_color': 'CCCCCC', 'contributors_enabled': False, 'url': 'https://t.co/hrcIziHrkf', 'is_translator': False, 'favourites_count': 3507, 'location': 'Brooklyn, NY', 'friends_count': 865, 'profile_image_url': 'http://pbs.twimg.com/profile_images/2623320981/kinlr53ma1flkp9jerk4_normal.jpeg', 'has_extended_profile': False, 'profile_background_tile': False, 'profile_link_color': '999999', 'description': 'MSR NYC. Machine learning, computational social science, fairness/accountability/transparency in ML. NIPS 2018 program chair, WiML co-founder, sloth enthusiast.', 'entities': {'url': {'urls': [{'display_url': 'dirichlet.net', 'url': 'https://t.co/hrcIziHrkf', 'expanded_url': 'http://dirichlet.net/', 'indices': [0, 23]}]}, 'description': {'urls': []}}, 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/2623320981/kinlr53ma1flkp9jerk4_normal.jpeg', 'default_profile': False, 'name': 'Hanna Wallach', 'profile_background_image_url_https': 'https://pbs.twimg.com/profile_background_images/521040468528754688/_Ayh3ZCE.jpeg', 'utc_offset': -14400, 'default_profile_image': False, 'id_str': '823957466'}, 'entities': {'urls': [{'display_url': 'twitter.com/i/web/status/9…', 'url': 'https://t.co/aB9Y5tTyHT', 'expanded_url': 'https://twitter.com/i/web/status/971171213216239616', 'indices': [117, 140]}], 'hashtags': [], 'user_mentions': [], 'symbols': []}, 'place': None, 'truncated': True, 'retweeted': False, 'id_str': '971171213216239616'}, 'entities': {'urls': [], 'hashtags': [], 'user_mentions': [{'name': 'Hanna Wallach', 'screen_name': 'hannawallach', 'indices': [3, 16], 'id': 823957466, 'id_str': '823957466'}], 'symbols': []}, 'place': None, 'truncated': False, 'retweeted': False, 'id_str': '986949027123154944'}

# +
# Print the text of the tweet
print(rt['text'])

# Print the text of tweet which has been retweeted
print(rt['retweeted_status']['text'])

# Print the user handle of the tweet
print(rt['user']['screen_name'])

# Print the user handle of the tweet which has been retweeted
print(rt['retweeted_status']['user']['screen_name'])
# -

# ## Processing Twitter text

# ### Tweet Items and Tweet Flattening

quoted_tweet = {'text': 'maybe if I quote tweet this lil guy https://t.co/BzbLDz9j6g', 'in_reply_to_status_id': None, 'source': '<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>', 'in_reply_to_status_id_str': None, 'is_quote_status': True, 'in_reply_to_screen_name': None, 'id': 989192330832891904, 'in_reply_to_user_id_str': None, 'in_reply_to_user_id': None, 'possibly_sensitive': False, 'lang': 'en', 'timestamp_ms': '1524676804632', 'created_at': 'Wed Apr 25 17:20:04 +0000 2018', 'quote_count': 0, 'contributors': None, 'coordinates': None, 'favorited': False, 'favorite_count': 0, 'retweet_count': 0, 'user': {'statuses_count': 71926, 'follow_request_sent': None, 'time_zone': 'Eastern Time (US & Canada)', 'profile_use_background_image': False, 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme16/bg.gif', 'screen_name': 'alexhanna', 'lang': 'en', 'friends_count': 2806, 'verified': False, 'profile_sidebar_border_color': '666666', 'profile_text_color': '333333', 'id': 661613, 'following': None, 'profile_sidebar_fill_color': 'CCCCCC', 'geo_enabled': True, 'created_at': 'Thu Jan 18 20:37:52 +0000 2007', 'notifications': None, 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/661613/1524231456', 'protected': False, 'listed_count': 246, 'profile_background_color': '000000', 'contributors_enabled': False, 'url': 'http://alex-hanna.com', 'is_translator': False, 'favourites_count': 23526, 'location': 'Toronto, ON', 'followers_count': 4275, 'profile_image_url': 'http://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'translator_type': 'regular', 'profile_background_tile': False, 'profile_link_color': '0671B8', 'description': 'Assistant professor @UofT. Protest, media, computation. Trans. Roller derby athlete @TOROLLERDERBY (Kate Silver #538). She/her.', 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'default_profile': False, 'name': 'Alex Hanna, Data Witch', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme16/bg.gif', 'utc_offset': -14400, 'default_profile_image': False, 'id_str': '661613'}, 'quoted_status_id': 989191655759663105, 'geo': None, 'filter_level': 'low', 'display_text_range': [0, 35], 'reply_count': 0, 'entities': {'urls': [{'display_url': 'twitter.com/alexhanna/stat…', 'url': 'https://t.co/BzbLDz9j6g', 'expanded_url': 'https://twitter.com/alexhanna/status/989191655759663105', 'indices': [36, 59]}], 'hashtags': [], 'user_mentions': [], 'symbols': []}, 'quoted_status': {'text': 'O 280 characters, 280 characters! Wherefore art thou 280 characters?\nDeny thy JSON and refuse thy key.\nOr, if thou… https://t.co/MlFg4qFnEC', 'in_reply_to_status_id': None, 'source': '<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>', 'in_reply_to_status_id_str': None, 'is_quote_status': False, 'in_reply_to_screen_name': None, 'id': 989191655759663105, 'in_reply_to_user_id_str': None, 'in_reply_to_user_id': None, 'lang': 'en', 'created_at': 'Wed Apr 25 17:17:23 +0000 2018', 'quote_count': 0, 'contributors': None, 'coordinates': None, 'favorited': False, 'favorite_count': 1, 'retweet_count': 0, 'user': {'statuses_count': 71925, 'follow_request_sent': None, 'time_zone': 'Eastern Time (US & Canada)', 'profile_use_background_image': False, 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme16/bg.gif', 'screen_name': 'alexhanna', 'lang': 'en', 'friends_count': 2806, 'verified': False, 'profile_sidebar_border_color': '666666', 'profile_text_color': '333333', 'id': 661613, 'following': None, 'profile_sidebar_fill_color': 'CCCCCC', 'geo_enabled': True, 'created_at': 'Thu Jan 18 20:37:52 +0000 2007', 'notifications': None, 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/661613/1524231456', 'protected': False, 'listed_count': 246, 'profile_background_color': '000000', 'contributors_enabled': False, 'url': 'http://alex-hanna.com', 'is_translator': False, 'favourites_count': 23526, 'location': 'Toronto, ON', 'followers_count': 4275, 'profile_image_url': 'http://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'translator_type': 'regular', 'profile_background_tile': False, 'profile_link_color': '0671B8', 'description': 'Assistant professor @UofT. Protest, media, computation. Trans. Roller derby athlete @TOROLLERDERBY (Kate Silver #538). She/her.', 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/980799823900180483/J9CDOX_X_normal.jpg', 'default_profile': False, 'name': 'Alex Hanna, Data Witch', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme16/bg.gif', 'utc_offset': -14400, 'default_profile_image': False, 'id_str': '661613'}, 'geo': None, 'extended_tweet': {'display_text_range': [0, 191], 'full_text': 'O 280 characters, 280 characters! Wherefore art thou 280 characters?\nDeny thy JSON and refuse thy key.\nOr, if thou wilt not, be but sworn my love,\nAnd I’ll no longer be a 140 character tweet.', 'entities': {'urls': [], 'hashtags': [], 'user_mentions': [], 'symbols': []}}, 'filter_level': 'low', 'reply_count': 1, 'entities': {'urls': [{'display_url': 'twitter.com/i/web/status/9…', 'url': 'https://t.co/MlFg4qFnEC', 'expanded_url': 'https://twitter.com/i/web/status/989191655759663105', 'indices': [116, 139]}], 'hashtags': [], 'user_mentions': [], 'symbols': []}, 'place': None, 'truncated': True, 'retweeted': False, 'id_str': '989191655759663105'}, 'place': None, 'truncated': False, 'retweeted': False, 'quoted_status_id_str': '989191655759663105', 'id_str': '989192330832891904'}

# +
# Print the tweet text
print(quoted_tweet['text'])

# Print the quoted tweet text
print(quoted_tweet['quoted_status']['text'])

# Print the quoted tweet's extended (140+) text
print(quoted_tweet['quoted_status']['extended_tweet']['full_text'])

# Print the quoted user location
print(quoted_tweet['quoted_status']['user']['location'])

# +
# Store the user screen_name in 'user-screen_name'
quoted_tweet['user-screen_name'] = quoted_tweet['user']['screen_name']

# Store the quoted_status text in 'quoted_status-text'
quoted_tweet['quoted_status-text'] = quoted_tweet['quoted_status']['text']

# Store the quoted tweet's extended (140+) text in 
# 'quoted_status-extended_tweet-full_text'
quoted_tweet['quoted_status-extended_tweet-full_text'] = quoted_tweet['quoted_status']['extended_tweet']['full_text']
# -

quoted_tweet


# ### A tweet flattening function

def flatten_tweets(tweets_json):
    """ Flattens out tweet dictionaries so relevant JSON
        is in a top-level dictionary."""
    tweets_list = []
    
    # Iterate through each tweet
    for tweet in tweets_json:
        tweet_obj = json.loads(tweet)
    
        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
        
        # Store the user location name in 'user-location'
        tweet_obj['user-location'] = tweet_obj['user']['location'] 
        
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] =tweet_obj['retweeted_status']['text']
            
        tweets_list.append(tweet_obj)
    return tweets_list


# ### Loading tweets into a DataFrame

# +
# Import pandas
import pandas as pd

# Flatten the tweets and store in `tweets`
tweets = flatten_tweets(data_science_json)

# Create a DataFrame from `tweets`
ds_tweets = pd.DataFrame(tweets)

# Print out the first 5 tweets from this dataset
print(ds_tweets['text'].values[0:5])
# -

# ### Finding keywords

# +
# Find mentions of #python in 'text'
python = ds_tweets['text'].str.contains('#python', case = False)

# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / len(ds_tweets))


# -

# ### Looking for text in all the wrong places

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


# ### Comparing #python to #rstats

# +
# Find mentions of #python in all text fields
python = check_word_in_tweet('#python', ds_tweets)

# Find mentions of #rstats in all text fields
rstats = check_word_in_tweet('#rstats', ds_tweets)

# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / ds_tweets.shape[0])

# Print proportion of tweets mentioning #rstats
print("Proportion of #rstats tweets:", np.sum(rstats) / ds_tweets.shape[0])
# -

# ### Creating time series data frame

# +
# Print created_at to see the original format of datetime in Twitter data
print(ds_tweets['created_at'].head())

# Convert the created_at column to np.datetime object
ds_tweets['created_at'] = pd.to_datetime(ds_tweets['created_at'])

# Print created_at to see new format
print(ds_tweets['created_at'].head())

# Set the index of ds_tweets to created_at
ds_tweets = ds_tweets.set_index('created_at')
# -

# ### Generating mean frequency

# +
# Create a python column
ds_tweets['python'] = check_word_in_tweet('#python', ds_tweets)

# Create an rstats column
ds_tweets['rstats'] = check_word_in_tweet('#rstats', ds_tweets)
# -

# ### Plotting mean frequency

# +
# Average of python column by day
mean_python = ds_tweets['python'].resample('D').mean()

# Average of rstats column by day
mean_rstats = ds_tweets['rstats'].resample('D').mean()

# Plot mean python by day(green)/mean rstats by day(blue)
plt.plot(mean_python.index.day, mean_python, color = 'green')
plt.plot(mean_rstats.index.day, mean_rstats, color = 'blue')

# Add labels and show
plt.xlabel('Day'); plt.ylabel('Frequency')
plt.title('Language mentions over time')
plt.legend(('#python', '#rstats'))
plt.show()
# -

# ### Loading VADER

# +
# Load SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Instantiate new SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Generate sentiment scores
sentiment_scores = ds_tweets['text'].apply(sid.polarity_scores)
# -

# ### Calculating sentiment scores

# +
# Print out the text of a positive tweet
print(ds_tweets[sentiment > 0.6]['text'].values)

# Print out the text of a negative tweet
print(ds_tweets[sentiment < 0.6]['text'].values)

# Generate average sentiment scores for #python
sentiment_py = sentiment[check_word_in_tweet('#python', ds_tweets)].resample('D').mean()

# Generate average sentiment scores for #rstats
sentiment_r = sentiment[check_word_in_tweet('#rstats', ds_tweets)].resample('D').mean()
# -

# ### Plotting sentiment scores

# +
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

# ## Twitter Networks

# ### Creating retweet network

# +
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

# ### Creating reply network

# +
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

# ### Visualizing retweet network

# +
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
plt.axis('off'); 
plt.show()
# -

# ### In-degree centrality

# +
column_names = ['screen_name', 'betweenness_centrality']

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

# ### Betweenness Centrality

# +
# Generate betweenness centrality for retweets 
rt_centrality = nx.betweenness_centrality(G_rt)

# Generate betweenness centrality for replies 
reply_centrality = nx.betweenness_centrality(G_reply)

# Store centralities in data frames
rt = pd.DataFrame(list(rt_centrality.items()), columns = column_names)
reply = pd.DataFrame(list(reply_centrality.items()), columns = column_names)

# Print first five results in descending order of centrality
print(rt.sort_values('betweenness_centrality', ascending = False).head())

# Print first five results in descending order of centrality
print(reply.sort_values('betweenness_centrality', ascending = False).head())
# -

# ### Ratios

# +
column_names = ['screen_name', 'degree']

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

# ## Putting Twitter data on the map

# ### Accessing user-defined location

# +
# Print out the location of a single tweet
print(tweet_json['user']['location'])

# Flatten and load the SOTU tweets into a dataframe
tweets_sotu = pd.DataFrame(flatten_tweets(tweets_sotu_json))

# Print out top five user-defined locations
print(tweets_sotu['user-location'].value_counts().head())


# -

# ### Accessing bounding box

# +
def getBoundingBox(place):
    """ Returns the bounding box coordinates."""
    return place['bounding_box']['coordinates']

# Apply the function which gets bounding box coordinates
bounding_boxes = tweets_sotu['place'].apply(getBoundingBox)

# Print out the first bounding box coordinates
print(bounding_boxes.values[0])


# -

# ### Calculating the centroid

# +
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

# ### Creating Basemap map

# +
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

# ### Plotting centroid coordinates

# +
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
m.scatter(lat, lon, latlon = True, alpha = 0.7)
plt.show()
# -

# ### Coloring by sentiment

# +
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
