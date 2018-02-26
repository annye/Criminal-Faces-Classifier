import theano
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import History
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import performance as perform
import sys
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.learning_curve import learning_curve

def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def get_traintest_validation_split(df):
    """ Split the data into a train_test dataset and a validation dataset
        train_test will later be split into the training and testing
        dataset
        Fitting using gradient descent etc is only performed on the
        training set, fitting via the act of hyperparameter optimisation
        uses the training and test dataset as tuners (this inherently fits
        the data to the training and test sets).
        This is the reason that the test dataset is not sufficient as a
        validation set
        (it can result in overestimated performance as the current optimised
         model has been fine-tuned to performed well on the test set)
         This is why the final model is tested on a hold-out validation set
    """
    train_test, validation = np.split(df.sample(frac=1),
                                      [int(.9 * len(df))])
    y = np.ravel(train_test['Class'])
    train_test = train_test.drop('Class', 1)
    x = train_test.as_matrix()
    return x, y, validation

def prepare_design_target(df):
    """ Prepare design matrix (X) and target vector (y)

    """
    y = np.ravel(df['Class'])
    x = df.drop('Class', 1)
    x = x.as_matrix()
    return x, y


def manual_ann2(X, y, validation, pca_dim):
    """ Keras Sequential model with Manual validation
    """
    # fix random seed for reproducibility
    seed = 13
    np.random.seed(seed)
    cvscores = []
    learning_rate = 0.2
    sgd = SGD(lr=learning_rate, momentum=0.05, nesterov=False)

    def create_baseline():
        # NN architecture
        input_nodes = pca_dim
        hidden1_nodes = 50
        hidden2_nodes = 50
        hidden3_nodes = 50
        output_nodes = 1
        dropout_rate = 0.1
        model = Sequential()
        model.add(Dense(output_dim=hidden1_nodes,
                        input_dim=input_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden2_nodes,
                        input_dim=hidden1_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden3_nodes,
                        input_dim=hidden2_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=output_nodes,
                        input_dim=hidden3_nodes,
                        activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline,
                                nb_epoch=100,
                                batch_size=200,
                                verbose=2)
    kfold = StratifiedKFold(n_splits=10,
                            shuffle=True,
                            random_state=seed)
    results = cross_val_score(estimator,
                              X,
                              y,
                              cv=kfold)
    
    print("Results of 10-fold Cross-Validation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    set_trace()

    title = "Learning Curves (Sequential Neural Net - Criminal men/Non-criminal women)"
    plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1)
    set_trace()
    
    accuracy = pd.DataFrame({'10 Fold Accuracy': [results.mean()*100]})
    accuracy.to_csv('C:\Thesis111217\CriminalClassifier\Outputs\All_images_acc.csv')

    # Plot Precision/Recall curve
    validation_y = np.ravel(validation['Class'])
    validation_x = validation.drop('Class', 1)
    validation_x = validation_x.as_matrix()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, random_state=42)
    model2 = create_baseline()
    model2.fit(X,y)
    pred = model2.predict(validation_x)
    pred = [item for sublist in pred for item in sublist]
    pred = [int(round(n, 0)) for n in pred]
    class_names = ['Non-Criminal', 'Criminal']
    cm = perform.get_confusion_matrix(validation_y, pred)
    perform.plot_confusion_matrix(cm, class_names)
    set_trace()
    print 'Classification report:\n {}'.format(classification_report(validation_y, pred))
    
    '''
    # Make predictions on hold-out validation set
    
    model = create_baseline()
    model.fit(X, y, epochs=100, batch_size=200, verbose=2)
    set_trace()
    pred = model.predict(validation_x)
    pred = [item for sublist in pred for item in sublist]
    pred = [int(round(n, 0)) for n in pred]
    class_names = ['Non-Criminal', 'Criminal']
    cm = perform.get_confusion_matrix(validation_y, pred)
    perform.plot_confusion_matrix(cm, class_names)
    print 'Classification report:\n {}'.format(classification_report(validation_y, pred))
    '''
    return 


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    #plt.savefig('C:\Thesis111217\CriminalClassifier\Outputs\LearningCurve_all.png')
    #plt.clf()

