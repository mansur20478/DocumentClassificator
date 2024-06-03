import argparse
from datasets import load_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tfslf import TfslfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # evaluation



SEED = 42


def main(args):
    X, y, y_labels = load_data(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    vectorizer = None
    vectorizer_name = None
    if args.vectorizer == 'tf-idf':
        vectorizer = TfidfVectorizer()
        vectorizer_name = 'TF-IDF'
    elif args.vectorizer == 'tf-slf':
        vectorizer = TfslfVectorizer()
        vectorizer_name = 'TF-SLF'
    else:
        raise ValueError(f'No {args.vectorizer} vectorizer')
    
    model = None
    model_name = None
    if args.model == 'MultinomialNB':
        model = MultinomialNB()
        model_name = 'Multinomial Naive Bayes'
    elif args.model == 'LinearSVC':
        model = LinearSVC()
        model_name = 'Linear Support Vector'
    elif args.model == 'RandomForest':
        n_estimators = 100
        model = RandomForestClassifier(n_estimators)
        model_name = 'Random Forest'
    elif args.model == 'KNN':
        model = KNeighborsClassifier(10)
        model_name = 'K-Nearest Neighbors'
    elif args.model == 'NearestCentroid':
        model = NearestCentroid()
        model_name = "Nearest Centroid"
    else:
        raise ValueError(f'No {args.model} method')
    

    # Prepare vectorizer and model
    X1_train = vectorizer.fit_transform(X_train, y_train)
    model.fit(X1_train, y_train)

    X1_test = vectorizer.transform(X_test)
    y_pred = model.predict(X1_test)
        
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f'Using dataset {args.dataset}')
    print(f'{model_name} with {vectorizer_name}:')
    print('-' * 40)
    print(f'f1: {f1:.4f}')
    print(f'accuracy: {accuracy:.4f}')
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-v', '--vectorizer')
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()

    main(args)