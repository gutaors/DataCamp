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

# + dc={"key": "3"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 1. Darwin's bibliography
# <p><img src="https://assets.datacamp.com/production/project_607/img/CharlesDarwin.jpg" alt="Charles Darwin" width="300px"></p>
# <p>Charles Darwin is one of the few universal figures of science. His most renowned work is without a doubt his "<em>On the Origin of Species</em>" published in 1859 which introduced the concept of natural selection. But Darwin wrote many other books on a wide range of topics, including geology, plants or his personal life. In this notebook, we will automatically detect how closely related his books are to each other.</p>
# <p>To this purpose, we will develop the bases of <strong>a content-based book recommendation system</strong>, which will determine which books are close to each other based on how similar the discussed topics are. The methods we will use are commonly used in text- or documents-heavy industries such as legal, tech or customer support to perform some common task such as text classification or handling search engine queries.</p>
# <p>Let's take a look at the books we'll use in our recommendation system.</p>

# + dc={"key": "3"} tags=["sample_code"]
# Import library
import glob

# The books files are contained in this folder
folder = "datasets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder + '*.txt')
files.sort()
files

# + dc={"key": "10"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 2. Load the contents of each book into Python
# <p>As a first step, we need to load the content of these books into Python and do some basic pre-processing to facilitate the downstream analyses. We call such a collection of texts <strong>a corpus</strong>. We will also store the titles for these books for future reference and print their respective length to get a gauge for their contents.</p>
# -

with open('datasets/Autobiography.txt', encoding = 'utf-8-sig') as f: 
    data = f.read()
    print(data)

# + dc={"key": "10"} tags=["sample_code"]
# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
txts = []
titles = []

for n in files:
    # Open each file
    with open(n, encoding = 'utf-8-sig') as f: 
    # Remove all non-alpha-numeric characters
        data = re.sub('[\W_]+', ' ', f.read())
    # Store the texts and titles of the books in two separate lists
        txts.append(data)
        titles.append(os.path.basename(n).replace('.txt', ''))

# Print the length, in characters, of each book
[len(t) for t in txts]

# + dc={"key": "17"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 3. Find "On the Origin of Species"
# <p>For the next parts of this analysis, we will often check the results returned by our method for a given book. For consistency, we will refer to Darwin's most famous book: "<em>On the Origin of Species</em>." Let's find to which index this book is associated.</p>

# + dc={"key": "17"} tags=["sample_code"]
for i in range(len(titles)):
    # Store the index if the title is "OriginofSpecies"
    if titles[i] == "OriginofSpecies":
        ori = i
        # Print the stored index
        print(ori)

# + dc={"key": "24"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 4. Tokenize the corpus
# <p>As a next step, we need to transform the corpus into a format that is easier to deal with for the downstream analyses. We will tokenize our corpus, i.e., transform each text into a list of the individual words (called tokens) it is made of. To check the output of our process, we will print the first 20 tokens of "<em>On the Origin of Species</em>".</p>

# + dc={"key": "24"} tags=["sample_code"]
# Define a list of stop words
stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())

# Convert the text to lower case 
txts_lower_case = [txt.lower() for txt in txts]

# Transform the text into tokens 
txts_split = [txt.split() for txt in txts_lower_case]

# Remove tokens which are part of the list of stop words
texts = [[word for word in txt if word not in stoplist] for txt in txts_split]

# Print the first 20 tokens for the "On the Origin of Species" book
texts[ori][:20]

# + dc={"key": "31"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 5. Stemming of the tokenized corpus
# <p>If you have read <em>On the Origin of Species</em>, you will have noticed that Charles Darwin can use different words to refer to a similar concept. For example, the concept of selection can be described by words such as <em>selection</em>, <em>selective</em>, <em>select</em> or <em>selects</em>. This will dilute the weight given to this concept in the book and potentially bias the results of the analysis.</p>
# <p>To solve this issue, it is a common practice to use a <strong>stemming process</strong>, which will group together the inflected forms of a word so they can be analysed as a single item: <strong>the stem</strong>. In our <em>On the Origin of Species</em> example, the words related to the concept of selection would be gathered under the <em>select</em> stem.</p>
# <p>As we are analysing 20 full books, the stemming algorithm can take several minutes to run and, in order to make the process faster, we will directly load the final results from a pickle file and review the method used to generate it.</p>

# +
# Load the Porter stemming function from the nltk package
from nltk.stem import PorterStemmer

# Create an instance of a PorterStemmer object
porter = PorterStemmer()

# For each token of each text, we generated its stem 
texts_stem = [[porter.stem(token) for token in text] for text in texts]

# Save to pickle file
pickle.dump(texts_stem, open("datasets/texts_stem_new.p", "wb"))

# Print the 20 first stemmed tokens from the "On the Origin of Species" book
texts_stem[ori][:20]
# -

texts_stem[0][0]


# + dc={"key": "38"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 6. Building a bag-of-words model
# <p>Now that we have transformed the texts into stemmed tokens, we need to build models that will be useable by downstream algorithms.</p>
# <p>First, we need to will create a universe of all words contained in our corpus of Charles Darwin's books, which we call <em>a dictionary</em>. Then, using the stemmed tokens and the dictionary, we will create <strong>bag-of-words models</strong> (BoW) of each of our texts. The BoW models will represent our books as a list of all uniques tokens they contain associated with their respective number of occurrences. </p>
# <p>To better understand the structure of such a model, we will print the five first elements of one of the "<em>On the Origin of Species</em>" BoW model.</p>

# + dc={"key": "38"} tags=["sample_code"]
# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(texts_stem)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [dictionary.doc2bow(txt) for txt in texts_stem]

# Print the first five elements of the On the Origin of species' BoW model
bows[ori][:5]

# + dc={"key": "45"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 7. The most common words of a given book
# <p>The results returned by the bag-of-words model is certainly easy to use for a computer but hard to interpret for a human. It is not straightforward to understand which stemmed tokens are present in a given book from Charles Darwin, and how many occurrences we can find.</p>
# <p>In order to better understand how the model has been generated and visualize its content, we will transform it into a DataFrame and display the 10 most common stems for the book "<em>On the Origin of Species</em>".</p>

# + dc={"key": "45"} tags=["sample_code"]
# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the BoW model for "On the Origin of Species" into a DataFrame
df_bow_origin = pd.DataFrame(bows[ori], columns= ['index', 'occurrences'])

# Add a column containing the token corresponding to the dictionary index
df_bow_origin['token'] = df_bow_origin['index'].apply(lambda x: dictionary[x])

# Sort the DataFrame by descending number of occurrences and print the first 10 values
df_bow_origin.sort_values('occurrences', ascending = False, inplace = True)
df_bow_origin.head()

# + dc={"key": "52"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 8. Build a tf-idf model
# <p>If it wasn't for the presence of the stem "<em>speci</em>", we would have a hard time to guess this BoW model comes from the <em>On the Origin of Species</em> book. The most recurring words are, apart from few exceptions, very common and unlikely to carry any information peculiar to the given book. We need to use an additional step in order to determine which tokens are the most specific to a book.</p>
# <p>To do so, we will use a <strong>tf-idf model</strong> (term frequency–inverse document frequency). This model defines the importance of each word depending on how frequent it is in this text and how infrequent it is in all the other documents. As a result, a high tf-idf score for a word will indicate that this word is specific to this text.</p>
# <p>After computing those scores, we will print the 10 words most specific to the "<em>On the Origin of Species</em>" book (i.e., the 10 words with the highest tf-idf score).</p>

# + dc={"key": "52"} tags=["sample_code"]
import operator 
# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
model = TfidfModel(bows)

# Print the model for "On the Origin of Species"
tf_idf = sorted([i for i in model[bows[ori]]], key = operator.itemgetter(1), reverse = True)[:10]
words = [dictionary[k] for (k, v) in tf_idf]
print(words)

# + dc={"key": "59"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 9. The results of the tf-idf model
# <p>Once again, the format of those results is hard to interpret for a human. Therefore, we will transform it into a more readable version and display the 10 most specific words for the "<em>On the Origin of Species</em>" book.</p>

# + dc={"key": "59"} tags=["sample_code"]
# Convert the tf-idf model for "On the Origin of Species" into a DataFrame
df_tfidf = pd.DataFrame(model[bows[ori]], columns = ['id', 'score'])

# Add the tokens corresponding to the numerical indices for better readability
df_tfidf['token'] = df_tfidf['id'].map(lambda x: dictionary[x])

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
df_tfidf.sort_values('score', ascending = False, inplace = True)

df_tfidf.head(10)

# + dc={"key": "66"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 10. Compute distance between texts
# <p>The results of the tf-idf algorithm now return stemmed tokens which are specific to each book. We can, for example, see that topics such as selection, breeding or domestication are defining "<em>On the Origin of Species</em>" (and yes, in this book, Charles Darwin talks quite a lot about pigeons too). Now that we have a model associating tokens to how specific they are to each book, we can measure how related to books are between each other.</p>
# <p>To this purpose, we will use a measure of similarity called <strong>cosine similarity</strong> and we will visualize the results as a distance matrix, i.e., a matrix showing all pairwise distances between Darwin's books.</p>

# + dc={"key": "66"} tags=["sample_code"]
# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
sims = similarities.MatrixSimilarity(model[bows])

# Transform the resulting list into a dataframe
sim_df = pd.DataFrame(list(sims), index = titles, columns = titles)

# Print the resulting matrix
sim_df

# + dc={"key": "73"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 11. The book most similar to "On the Origin of Species"
# <p>We now have a matrix containing all the similarity measures between any pair of books from Charles Darwin! We can now use this matrix to quickly extract the information we need, i.e., the distance between one book and one or several others. </p>
# <p>As a first step, we will display which books are the most similar to "<em>On the Origin of Species</em>," more specifically we will produce a bar chart showing all books ranked by how similar they are to Darwin's landmark work.</p>

# + dc={"key": "73"} tags=["sample_code"]
# This is needed to display plots in a notebook
# %matplotlib inline

# Import libraries
import matplotlib.pyplot as plt

# Select the column corresponding to "On the Origin of Species" and 
v = sim_df.loc[:, 'OriginofSpecies']

# Sort by ascending scores
v_sorted = v.sort_values(ascending = False)

# Plot this data has a horizontal bar plot
v_sorted.plot(kind = 'bar', figsize = (16, 10));

# Modify the axes labels and plot title for a better readability
plt.title('The Similarity between "On the Origin of Species" and other books from Darwin')
plt.xlabel("Book Title")
plt.ylabel("Similary Score");
# plt.xticks(rotation = 70)

# + dc={"key": "80"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 12. Which books have similar content?
# <p>This turns out to be extremely useful if we want to determine a given book's most similar work. For example, we have just seen that if you enjoyed "<em>On the Origin of Species</em>," you can read books discussing similar concepts such as "<em>The Variation of Animals and Plants under Domestication</em>" or "<em>The Descent of Man, and Selection in Relation to Sex</em>." If you are familiar with Darwin's work, these suggestions will likely seem natural to you. Indeed, <em>On the Origin of Species</em> has a whole chapter about domestication and <em>The Descent of Man, and Selection in Relation to Sex</em> applies the theory of natural selection to human evolution. Hence, the results make sense.</p>
# <p>However, we now want to have a better understanding of the big picture and see how Darwin's books are generally related to each other (in terms of topics discussed). To this purpose, we will represent the whole similarity matrix as a dendrogram, which is a standard tool to display such data. <strong>This last approach will display all the information about book similarities at once.</strong> For example, we can find a book's closest relative but, also, we can visualize which groups of books have similar topics (e.g., the cluster about Charles Darwin personal life with his autobiography and letters). If you are familiar with Darwin's bibliography, the results should not surprise you too much, which indicates the method gives good results. Otherwise, next time you read one of the author's book, you will know which other books to read next in order to learn more about the topics it addressed.</p>

# + dc={"key": "80"} tags=["sample_code"]
# Import libraries
from scipy.cluster import hierarchy

# Compute the clusters from the similarity matrix,
# using the Ward variance minimization algorithm
Z = hierarchy.linkage(sim_df, 'ward')

# Display this result as a horizontal dendrogram
hierarchy.dendrogram(Z, leaf_font_size=8, labels=sim_df.index, orientation="left");
# -


