import glob as gb
from shutil import copyfile
import sys


def set_trace():
    """A Poor mans break point
       without this in iPython debugger can generate strange characters.
    """
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def imageWithEmotionExtraction():
    emotions = ["neutral",
                "anger",
                "contempt",
                "disgust",
                "fear",
                "happy",
                "sadness",
                "surprise"]

    # Returns a list of all folders with participant numbers
    participants = gb.glob("source_emotion\\*")

    for x in participants:
        # Get the last 4 characters which is the participant number
        part = "%s" % x[-4:]
        for sessions in gb.glob("%s\\*" % x):
            for files in gb.glob("%s\\*" % sessions):
                current_session = files[20:23]
                file = open(files, 'r')
                emotion = int(float(file.readline()))
                # get path for last image in sequence, which contains the emotion
                sourcefile_emotion = \
                    gb.glob("source_images\\%s\\%s\\*" % (part,
                                                          current_session))[-1]
                # Get path for first image which is the neutral image
                sourcefile_neutral = \
                    gb.glob("source_images\\%s\\%s\\*" % (part,
                                                          current_session))[0]
                # Generate path to put emotion image
                dest_emot = \
                    "sorted_set\\%s\\%s" % (emotions[emotion],
                                            sourcefile_emotion[25:])
                # Generate path for neutral image
                dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]

                # Copy files
                copyfile(sourcefile_emotion, dest_emot)
                copyfile(sourcefile_neutral, dest_neut)


def main():
    imageWithEmotionExtraction()
    set_trace()


if __name__ == '__main__':
    imageWithEmotionExtraction()
