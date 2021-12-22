# Design a Recommendation Engine 

## Project Overview
I  analyzed the interactions that users had with articles on the IBM Watson Studio platform, and designed a recommendation engine on new articles that users might like.

## Project Steps

I. Exploratory Data Analysis

Before making recommendations of any kind, I explored the data available on the IBM Watson Studio platform.

II. Rank Based Recommendations

To get started in building recommendations, I first found the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

III. User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, I looked at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This is a step in the right direction towards more personal recommendations for the users.

IV. Matrix Factorization

Finally, I completed a machine learning approach to building recommendations. Using the user-item interactions, I built out a matrix decomposition. Using the decomposition, I got an idea of how well I could predict new articles an individual might interact with. I then discussed which methods to use moving forward, and how to test how well the recommendations are working for engaging users.
