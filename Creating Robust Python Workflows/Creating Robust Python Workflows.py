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

# +
### MODULE 1

### Python Programming Principles

# + active=""
# In this chapter, we will discuss three principles that guide decisions made by Python programmers. You will put these principles into practice in the coding exercises and throughout the rest of the course!

# +
### Functions and iteration

def print_files(filenames):
    # Set up the loop iteration instructions
    for name in filenames:
        # Use pathlib.Path to print out each file
        print(Path(name).read_text())
        
def list_files(filenames):
    # Use pathlib.Path to read the contents of each file
    return [Path(name).read_text()
            # Obtain each name from the list of filenames
            for name in filenames]

filenames = 'diabetes.txt', 'boston.txt', 'digits.txt', 'iris.txt', 'wine.txt'
print_files(filenames)
file_contents = list_files(filenames)
# -



# +
### Find matches

def get_matches(filename, query):
    # Obtain lines from the input file
    return [line for line in Path(filename).open()
            # Filter the list comprehension
            if query in line]

# Iterate over files to find all matching lines
matches = [get_matches(name, 'Number of') for name in filenames]
print(matches)
# -



# +
### Dataset dimensions

def flatten(nested_list):
    return (element 
            # Obtain each list from the list of lists
            for sublist in nested_list
            # Obtain each string from every list
            for element in sublist)

number_generator = (int(substring) for string in flatten(matches)
                    for substring in string.split() if substring.isdigit())
print(dict(zip(filenames, zip(number_generator, number_generator))))
# -



# +
### Extract words

def obtain_words(string):
    # Replace non-alphabetic characters with spaces
    return ''.join(char if char.isalpha() else ' ' for char in string).split()

def filter_words(words, minimum_length=3):
    # Remove words shorter than 3 characters
    return [word for word in words if len(word) >= minimum_length]

# Use a string method to convert the text to lowercase
words = obtain_words(Path('diabetes.txt').read_text().lower())
filtered_words = filter_words(words)

print(filtered_words)
# -



# +
### Most frequent words

def count_words(word_list):
    # Count the words in the input list
    return {word: word_list.count(word) for word in word_list}

# Create the dictionary of words and word counts
word_count_dictionary = count_words(filtered_words)

(pd.DataFrame(word_count_dictionary.items())
 .sort_values(by=1, ascending=False)
 .head()
 .plot(x=0, kind='barh', xticks=range(5), legend=False)
 .set_ylabel("")
)
plt.show()
# -



# +
### Instance method

# Fill in the first parameter in the pair_plot() definition
def pair_plot(self, vars=range(3), hue=None):
    return pairplot(pd.DataFrame(self.data), 
                    vars=vars, 
                    hue=hue, 
                    kind='reg')

ScikitData.pair_plot = pair_plot

# Create the diabetes instance of the ScikitData class
diabetes = ScikitData('diabetes')

# Call pairplot() to plot diabetes dataset variables
diabetes.pair_plot(vars=range(2, 6), hue=1)._legend.remove()
plt.show()
# -



# +
### Class method

# Fill in the decorator for the get_generator() definition
@classmethod
# Add the first parameter to the get_generator() definition
def get_generator(cls, names):
    return (cls(x) for x in names)

ScikitData.get_generator = get_generator

# Create a generator for the diabetes and iris datasets
dataset_generator = ScikitData.get_generator(['diabetes', 'iris'])

# Iterate over each dataset in the dataset generator
for dataset in dataset_generator:
    dataset.pair_plot()
    plt.show()
# -



# +
### MODULE 2 

### Documentation and Tests

# + active=""
# Documentation and tests are often overlooked, despite being essential to the success of all projects. In this chapter, you will learn how to include documentation in our code and practice Test-Driven Development (TDD), a process that puts tests first!
# -



# +
### TextFile hints

class TextFile:
  	# Add type hints to TextFile's __init__() method
    def __init__(self, name: str) -> None:
        self.lines = read_file(name).split('\n')

	# Add type hints to TextFile's get_list() method
    def get_lines(self) -> List[str]:
        return self.lines

help(TextFile)
# -



# +
### MatchFinder hints

class MatchFinder:
  	# Add type hints to the __init__()'s strings argument
    def __init__(self, strings: List[str]) -> None:
        self.strings = strings

	# Type annotate the query argument and return value
    def get_matches(self, query: Optional[str] = None) -> List[str]:
        return list(filter(lambda x: match in x, self.strings) if match else self.strings)

help(MatchFinder)
# -
















