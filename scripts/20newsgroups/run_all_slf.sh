DATASET=fetch_20newsgroups
VECTORIZER=tf-slf

python main.py -m MultinomialNB -d $DATASET -v $VECTORIZER
python main.py -m RandomForest -d $DATASET -v $VECTORIZER
python main.py -m LinearSVC -d $DATASET -v $VECTORIZER
python main.py -m KNN -d $DATASET -v $VECTORIZER
python main.py -m NearestCentroid -d $DATASET -v $VECTORIZER