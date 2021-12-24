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

# + dc={"key": "1d0b086e6c"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 1. Introduction
# <p>Everyone loves Lego (unless you ever stepped on one). Did you know by the way that "Lego" was derived from the Danish phrase leg godt, which means "play well"? Unless you speak Danish, probably not. </p>
# <p>In this project, we will analyze a fascinating dataset on every single lego block that has ever been built!</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/lego-bricks.jpeg" alt="lego"></p>

# + dc={"key": "044b2cef41"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 2. Reading Data
# <p>A comprehensive database of lego blocks is provided by <a href="https://rebrickable.com/downloads/">Rebrickable</a>. The data is available as csv files and the schema is shown below.</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/downloads_schema.png" alt="schema"></p>
# <p>Let us start by reading in the colors data to get a sense of the diversity of lego sets!</p>

# + dc={"key": "044b2cef41"} tags=["sample_code"]
# Import modules
import pandas as pd

# Read colors data
colors = pd.read_csv('datasets/colors.csv')

# Print the first few rows
colors.head()

# + dc={"key": "15c1e2ce38"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 3. Exploring Colors
# <p>Now that we have read the <code>colors</code> data, we can start exploring it! Let us start by understanding the number of colors available.</p>

# + dc={"key": "15c1e2ce38"} tags=["sample_code"]
# How many distinct colors are available?
num_colors = colors.shape[0]
print(num_colors)

# + dc={"key": "a5723ae5c2"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 4. Transparent Colors in Lego Sets
# <p>The <code>colors</code> data has a column named <code>is_trans</code> that indicates whether a color is transparent or not. It would be interesting to explore the distribution of transparent vs. non-transparent colors.</p>

# + dc={"key": "a5723ae5c2"} tags=["sample_code"]
# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby('is_trans')[['id', 'name', 'rgb']].count()
print(colors_summary)

# + dc={"key": "c9d0e58653"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 5. Explore Lego Sets
# <p>Another interesting dataset available in this database is the <code>sets</code> data. It contains a comprehensive list of sets over the years and the number of parts that each of these sets contained. </p>
# <p><img src="https://imgur.com/1k4PoXs.png" alt="sets_data"></p>
# <p>Let us use this data to explore how the average number of parts in Lego sets has varied over the years.</p>

# + dc={"key": "c9d0e58653"} tags=["sample_code"]
# %matplotlib inline
# Read sets data as `sets`
sets = pd.read_csv('datasets/sets.csv')
# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets.groupby('year')['num_parts'].mean()
# Plot trends in average number of parts by year
parts_by_year.plot()

# + dc={"key": "266a3f390c"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 6. Lego Themes Over Years
# <p>Lego blocks ship under multiple <a href="https://shop.lego.com/en-US/Themes">themes</a>. Let us try to get a sense of how the number of themes shipped has varied over the years.</p>

# + dc={"key": "266a3f390c"} tags=["sample_code"]
# themes_by_year: Number of themes shipped by year
themes_by_year = sets.groupby('year')['theme_id'].nunique()
themes_by_year.head()

# + dc={"key": "a293e5076e"} deletable=false editable=false run_control={"frozen": true} tags=["context"]
# ## 7. Wrapping It All Up!
# <p>Lego blocks offer an unlimited amount of fun across ages. We explored some interesting trends around colors, parts, and themes. </p>
# -


