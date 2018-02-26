
import cv2
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns


def set_trace():
    """A Poor mans break point without this in iPython debugger.

    Can generate strange characters.
    """
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


# Emotion list
emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]
# Initialize fisher face classifier
# fishface = cv2.face.FisherFaceRecognizer_create()
fishface = cv2.face.LBPHFaceRecognizer_create()
# fishface = cv2.face.EigenFaceRecognizer_create()
data = {}


def get_files(emotion):
    """Define function to get file list randomly shuffle it and split 80/20."""
    files = glob.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    # get first 80% of file list
    training = files[:int(len(files) * 0.8)]
    # get last 20% of file list
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


def make_sets():
    """Allocate sets for training and test."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            # open image
            image = cv2.imread(item)
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # append image array to training data list
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        # repeat above process for prediction set
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    """Run recognizer."""
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print "training fisher face classifier"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    print "predicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect)), fishface


def map_emotions(df):
    """Map emotions to code."""
    emotion_map = {0: 'neutral',
                   1: 'anger',
                   2: 'contempt',
                   3: 'disgust',
                   4: 'fear',
                   5: 'happy',
                   6: 'sadness',
                   7: 'surprise'}
    df['Emotion'] = df['Emotion Code'].map(emotion_map)
    return df


def plot_histographic_emotion_analysis(df, group):
    """Countplots."""
    try:
        sns.countplot(x='Emotion', data=df)
    except:
        set_trace()
    # plt.savefig('{}.png'.format(group))
    plt.show()
    # plt.clf()


def predict_emotions(files, recognizer, group):
    """Predict Emotions and analyse distribution."""
    pred_list = []
    
    
    for file in files:
        print group, file
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            pred, conf = recognizer.predict(gray)
        except:
            set_trace()
        tup = (file[-14:], pred, conf)
        pred_list.append(tup)
    pred_df = pd.DataFrame(pred_list, columns=['Image',
                                               'Emotion Code',
                                               'Confidence'])
    pred_df = map_emotions(pred_df)
    plot_histographic_emotion_analysis(pred_df, group)


def main():
    """Main function."""
    metascore = []
    for i in range(0, 1):
        correct, fishface = run_recognizer()
        print "got", correct, "percent correct!"
        metascore.append(correct)
    accuracy = pd.DataFrame({'Emotion Detector Accuracy': [np.mean(metascore)]})
    accuracy.to_csv('C:\Thesis111217\EmotionDetector\Outputs\EmotionDetector_acc.csv')
    print "\n\nend score:", np.mean(metascore), "percent correct!"
    set_trace()
    files = glob.glob("C:\Users\Creative\Documents\FacialExpressions\\Men_non_criminal_108\*")
    predict_emotions(files, fishface, 'Outputs\\Men_non_criminal_108')
    files = glob.glob("C:\Thesis111217\CriminalClassifier\static\\Women_criminal_108\*")
    predict_emotions(files, fishface, 'Outputs\\Women_criminal_108')


if __name__ == '__main__':
    main()
