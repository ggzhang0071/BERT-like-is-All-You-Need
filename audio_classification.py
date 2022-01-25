from gettext import install
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
import itertools
#from IPython.display import display

from tensorflow.keras.preprocessing import sequence

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

def display_results(y_test, pred_probs, cm=True):
    pred = np.argmax(pred_probs, axis=-1)
    one_hot_true = one_hot_encoder(y_test, len(pred), len(emotion_dict))
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    if cm:
        plot_confusion_matrix(confusion_matrix(y_test, pred), classes=emo_keys)

def convert_tokens_str_to_array(x_train):
    x_tokens_new=[]
    x_tokens= x_train=x_train.iloc[:,-1].tolist()
    for i in range(len(x_tokens)):
        tmp=[]
        for j in range(len(x_tokens[i])):
            if x_tokens[i][j].isdigit() and x_tokens[i][j]!=' ':
                tmp.append(int(x_tokens[i][j]))
        x_tokens_new.append(tmp)
                
    x_tokens = sequence.pad_sequences(x_tokens_new) 
    return  x_tokens

x_train = pd.read_csv('data/s2e/audio_train.csv')
x_test = pd.read_csv('data/s2e/audio_test.csv')

print(x_train.shape)
y_train = x_train['label']
y_test = x_test['label']

print(x_train.shape, x_test.shape)
cl_weight = dict(pd.Series(x_train['label']).value_counts(normalize=True))
print(dict(pd.Series(x_train['label']).value_counts()))

del x_train['label']
del x_test['label']
del x_train['wav_file']
del x_test['wav_file']
emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'fea': 3,
                'sur': 4,
                'neu': 5}

emo_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])

rf_classifier = RandomForestClassifier(n_estimators=1200, min_samples_split=25)

x_train=x_train.drop(labels="tokens",axis=1)
x_test=x_test.drop(labels="tokens",axis=1)


"""x_train=convert_tokens_str_to_array(x_train)
x_test=convert_tokens_str_to_array(x_test)"""


"""rf_classifier.fit(x_train, y_train)
# Predict

pred_probs = rf_classifier.predict_proba(x_test)

# Results
display_results(y_test, pred_probs)

with open('pred_probas/rf_classifier.pkl', 'wb') as f:
    pickle.dump(pred_probs, f)
"""


## TPOP
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from imblearn.combine import SMOTETomek
from  sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
smt = SMOTETomek(sampling_strategy='auto')


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
#model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
model = TPOTClassifier(generations=1, population_size=1, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)

# perform the search

x_train, y_train = smt.fit_resample(x_train, y_train)
model.fit(x_train, y_train)
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.7000000000000001, min_samples_leaf=15, min_samples_split=10, n_estimators=100, subsample=0.9000000000000001)
)
results = exported_pipeline.predict(x_test)
print("the results is:",results)

# export the best model
model.export('tpot_sonar_best_model.py')


"""##  autogluon 
from autogluon.tabular import TabularPredictor, TabularDataset
from imblearn.combine import SMOTETomek
smt = SMOTETomek(ratio='minority')

x_train = pd.read_csv('data/s2e/audio_train.csv')
x_test = pd.read_csv('data/s2e/audio_test.csv')

print(x_train.shape)
y_train = x_train['label']
y_test = x_test['label']

x_train = TabularDataset("data/s2e/audio_train.csv")
# fit the model
x_train, y_train = smt.fit_sample(x_train, y_train)

predictor = TabularPredictor(label="label").fit(x_train)
# make predictions on new data
x_test = TabularDataset("data/s2e/audio_test.csv")
prediction = predictor.predict(x_test)"""

## auto-pytorch
"""from autoPyTorch.api.tabular_classification import TabularClassificationTask
api = TabularClassificationTask()

api.search(
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    optimize_metric='accuracy',
    total_walltime_limit=500,
    func_eval_time_limit_secs=100
)

# Calculate test accuracy
y_pred = api.predict(x_test)
score = api.score(y_pred, y_test)
print("Accuracy score", score)"""



