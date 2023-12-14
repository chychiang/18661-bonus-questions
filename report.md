# ECE 18661 Bonus Questions

- [ECE 18661 Bonus Questions](#ece-18661-bonus-questions)
  - [Question 1: Decision Tree for Spotify Data](#question-1-decision-tree-for-spotify-data)
    - [1.1 Import Data](#11-import-data)
      - [Import the data into a Pandas dataframe](#import-the-data-into-a-pandas-dataframe)
      - [Of the remaining features which you believe may be useful for classification, which feature(s) do you estimate will be the most important? Which feature(s) will be the least important?](#of-the-remaining-features-which-you-believe-may-be-useful-for-classification-which-features-do-you-estimate-will-be-the-most-important-which-features-will-be-the-least-important)

## Question 1: Decision Tree for Spotify Data

### 1.1 Import Data

#### Import the data into a Pandas dataframe

Pandas is a data analysis library that is very useful for machine learning projects. Examine the data. Which features, if any, appear to not be useful for classification and should be removed? Print the final list of the feature names that you believe to be
useful.

```python
def load_data():
    """ Load Data Set, making sure to import the index column correctly
        Arguments:
            None
        Returns:
            Training data dataframe, training labels, testing data dataframe,
            testing labels, features list
    """
    df = pd.read_csv('spotify_data.csv', index_col=0, header=0)
    df = df.drop(['song_title', 'artist'], axis=1)
    df_without_target = df.drop(['target'], axis=1)
    labels = df['target']
    corr_w_target = df.corr()['target'].sort_values(ascending=False)
    print("="*80)
    print("Feature Correlation with Target")
    print(corr_w_target)
    
    plt.bar(corr_w_target.index, corr_w_target)
    plt.xticks(rotation=45)
    plt.title('Correlation with Target')
    plt.ylabel('Correlation')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig('corr_w_target.png')
    plt.clf()

    X_train, X_test, y_train, y_test = train_test_split(df_without_target, labels, test_size=0.2)
    return X_train, X_test, y_train, y_test, df_without_target.columns.values.tolist()
```

Prints out the following result:

```text
Dataframe Feature Names
['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'target', 'song_title', 'artist']
```

The columns that are clearly not useful for classification are:

1. song_title
2. artist

Thus, we drop them from the dataframe and print out the final list of feature names that I believe to be useful

```text
Dropped Dataframe Feature Names
['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'target']
```

#### Of the remaining features which you believe may be useful for classification, which feature(s) do you estimate will be the most important? Which feature(s) will be the least important?

Briefly explain your answers.
