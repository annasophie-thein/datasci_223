"""
Title: <DATASCI 223 exercise 2>
Short description: <This script will answer exercise 2>
Author: <Anna-Sophie Thein>
Created: <January 26, 2024>
Input:
Output:

Other requirements:
In terminal: python3 -m pip install pandas numpy matplotlib emnist seaborn jupyter
"""

# Import packages
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emnist
from hashlib import sha1

# Load the data, and reshape it into a 28x28 array

# The size of each image is 28x28 pixels
size = 28

# Extract the training split as images and labels
image, label = emnist.extract_training_samples('byclass')

# Add columns for each pixel value (28x28 = 784 columns)
raw_train = pd.DataFrame()

# Add a column showing the label
raw_train['label'] = label

# Add a column with the image data as a 28x28 array
raw_train['image'] = list(image)

# Repeat for the test split
image, label = emnist.extract_test_samples('byclass')
raw_test = pd.DataFrame()
raw_test['label'] = label
raw_test['image'] = list(image)

# Get the first row for each label
firsts = raw_train.groupby('label').first().reset_index()

# Now let's mess up the data a bit

# Percent of the time something dirty happens (0.1%) for each method
pct = 0.001

# Copy the splits into new dataframes to mess up
dirty_train = raw_train.copy()
dirty_test  = raw_test.copy()

# Add a column for a hash of the images (should make it easier to compare them)
dirty_train['image_hash'] = dirty_train['image'].apply(lambda x: sha1(x.tobytes()).hexdigest())
dirty_test['image_hash']  =  dirty_test['image'].apply(lambda x: sha1(x.tobytes()).hexdigest())

# For each row, 0.1% of the time, duplicate the row
dirty_train = pd.concat([dirty_train, dirty_train.sample(frac=pct)])
dirty_test  = pd.concat([dirty_test, dirty_test.sample(frac=pct)])

# For each row, 0.1% of the time, zero out the image array
dirty_train['image'] = dirty_train['image'].apply(lambda x: np.zeros((size, size)) if np.random.rand() < pct else x)
dirty_test['image']  =  dirty_test['image'].apply(lambda x: np.zeros((size, size)) if np.random.rand() < pct else x)

# Add a column for classification scores from a previous model
dirty_train['predict'] = np.random.normal(0.75, 0.1, dirty_train.shape[0])
dirty_test['predict']  = np.random.normal(0.75, 0.1, dirty_test.shape[0])

# For each row, 0.1% of the time, replace the predict column with a normal distribution centered on 0.25
dirty_train['predict'] = dirty_train['predict'].apply(lambda x: np.random.normal(0.25, 0.1) if np.random.rand() < pct else x)
dirty_test['predict']  =  dirty_test['predict'].apply(lambda x: np.random.normal(0.25, 0.1) if np.random.rand() < pct else x)

# For each row, 0.1% of the time, add/subtract 1 to the predict column
dirty_train['predict'] = dirty_train['predict'].apply(lambda x: x + 1 if np.random.rand() < pct/2 else x)
dirty_test['predict']  =  dirty_test['predict'].apply(lambda x: x + 1 if np.random.rand() < pct/2 else x)
dirty_train['predict'] = dirty_train['predict'].apply(lambda x: x - 1 if np.random.rand() < pct/2 else x)
dirty_test['predict']  =  dirty_test['predict'].apply(lambda x: x - 1 if np.random.rand() < pct/2 else x)

# For each row, 0.1% of the time, choose a column at random and set it to NaN
dirty_train = dirty_train.apply(lambda row: row if np.random.rand() > pct else row.apply(lambda x: np.nan if np.random.rand() > 0.5 else x), axis=1)
dirty_test  =  dirty_test.apply(lambda row: row if np.random.rand() > pct else row.apply(lambda x: np.nan if np.random.rand() > 0.5 else x), axis=1)

# Mislabel 0.1% of the data with strings that look like labels or missing values (e.g., names like "one", numbers greater than 62, " ")
# Create a list of bad labels
bad_labels = ['number', 'letter', 'maybe three?', 'think this is a seven', ' ', '', 'NaaN', 'Null'] + list(range(63, 100))
# For 0.1% of the rows, randomly choose a bad label
dirty_train['label'] = dirty_train['label'].apply(lambda x: np.random.choice(bad_labels) if np.random.rand() < pct else x)
dirty_test['label']  =  dirty_test['label'].apply(lambda x: np.random.choice(bad_labels) if np.random.rand() < pct else x)

# Add a column showing which split (train vs test) each row came from
raw_train['split'] = 'train'
raw_test['split']  = 'test'
dirty_train['split'] = 'train'
dirty_test['split'] = 'test'

# Let's start cleaning!

# Merge data into single dataframe
dirty_merged = merged = pd.concat([dirty_test, dirty_train], axis=0)

# Explore data
dirty_merged.info()
dirty_merged.describe
dirty_merged['label'].unique()

### Problem 1: Labels
work_df = dirty_merged.copy() # create copy of original dataframe
work_df["label"].replace(["NaaN", "Null","number","letter","nan",' ',''], np.nan, inplace=True) # replace bad labels with NaN
work_df['label'].unique()
work_df['label'] = work_df['label'].astype(str)
work_df = work_df[work_df['label'].astype(str).str.contains(r'\d', na=False)] # drop if label is text
work_df['label'] = work_df['label'].astype(float).astype(int) # convert to integer
work_df = work_df[work_df['label'] <= 61.0] # drop if above 61.0
work_df['label'].describe()
# Replace labels from manual dictionary for easier reading
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

work_df['label'] = work_df['label'].map(lambda x: LABELS[int(x)])

### Problem 2: Missing Values
work_df.dropna(axis=0, how='any', subset=None, inplace=False) # drop any rows with null values

### Problem 3: Duplicates
work_df[work_df.duplicated(subset='image_hash', keep=False)] # Find duplicates
work_df = work_df.drop_duplicates(subset='image_hash', keep='first') # Drop duplicates

### Problem 4: Zeroed image arrays
work_df[work_df['image'].apply(lambda x: (np.array(x) == 0).all())] # see if any images are all zeros
work_df = work_df[~work_df['image'].apply(lambda x: (np.array(x) == 0).all())] # drop if any images are all zeros

### Problem 5: out of bounds values for 'predict'
work_df['predict'].describe()
dirty_merged['predict'].describe()
# out of bounds values are chosen based on min and max of 'predict' before messing with the dataset
work_df = work_df[(work_df['predict'] >= 0.3) & (work_df['predict'] <= 1.29)] # drop values below 0.3 or above 1.29
work_df['predict'].value_counts(bins = 20, sort = False) # check distribution