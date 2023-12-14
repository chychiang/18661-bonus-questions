"""
Code to build decision tree classifier for spotify data
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def load_data():
    """ Load Data Set, making sure to import the index column correctly
        Arguments:
            None
        Returns:
            Training data dataframe, training labels, testing data dataframe,
            testing labels, features list
    """
    df = pd.read_csv('spotify_data.csv', index_col=0, header=0)
    print("="*80)
    print("Dataframe Feature Names")
    print(df.columns.values.tolist())
    df = df.drop(['song_title', 'artist'], axis=1)
    print("="*80)
    print("Dropped Dataframe Feature Names")
    print(df.columns.values.tolist())
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


def cv_grid_search(training_table, training_labels):
    """ Run grid search with cross-validation to try different
    hyperparameters
        Arguments:
            Training data dataframe and training labels
        Returns:
            Dictionary of best hyperparameters found by a grid search with
            cross-validation
    """
    tree = DecisionTreeClassifier()
    params = {'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [2, 4, 6, 8, 10],
                'class_weight': ['balanced', None]}
    gcv = GridSearchCV(estimator=tree, param_grid=params, cv=5)
    res = gcv.fit(training_table, training_labels)
    return res.best_params_


def plot_confusion_matrix(test_labels, pred_labels):
    """Plot confusion matrix
        Arguments:
            ground truth labels and predicted labels
        Returns:
            Writes image file of confusion matrix
    """
    cm = metrics.confusion_matrix(test_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['hate', 'love'])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.clf()

def graph_tree(model, training_features, class_names):
    """ Plot the tree of the trained model
        Arguments:
            Trained model, list of features, class names
        Returns:
            Writes PDF file showing decision tree representation
    """
    dot_data = export_graphviz(model, feature_names=training_features, class_names=class_names, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('tree')


def print_results(predictions, test_y):
    """Print results
        Arguments:
            Ground truth labels and predicted labels
        Returns:
            Prints precision, recall, F1-score, and accuracy
    """
    print("="*80)
    print("Best Model Metrics")
    print("Precision:", precision_score(test_y, predictions))
    print("Recall:", recall_score(test_y, predictions))
    print("Accuracy:", accuracy_score(test_y, predictions))
    print("F1:", f1_score(test_y, predictions))


def print_feature_importance(model, features):
    """Print feature importance
        Arguments:
            Trained model and list of features
        Returns:
            Prints ordered list of features, starting with most important,
            along with their relative importance (percentage).
    """
    print('='*80)
    print("Feature Importance")
    sort_index = np.argsort(model.feature_importances_)[::-1]
    for i in sort_index:
        print(features[i], model.feature_importances_[i])


def main():
    """Run the program"""
    # Load data
    train_x, test_x, train_y, test_y, features = load_data()

    # Cross Validation Training
    params = cv_grid_search(train_x, train_y)
    print("="*80)
    print("Best parameters:")
    for k, v in params.items():
        print(k, v)

    # Train and test model using hyperparameters
    best_model = DecisionTreeClassifier()
    best_model.fit(train_x, train_y)
    predictions = best_model.predict(test_x)

    # Confusion Matrix
    plot_confusion_matrix(test_y, list(predictions))

    # Graph Tree
    graph_tree(best_model, features, ['hate', 'love'])

    # Accuracy, Precision, Recall, F1
    print_results(predictions, test_y)

    # Feature Importance
    print_feature_importance(best_model, features)
    print("="*80)


if __name__ == '__main__':
    main()
