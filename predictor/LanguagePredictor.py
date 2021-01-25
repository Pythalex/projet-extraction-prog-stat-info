from langdetect import detect
from enum import Enum
from statistics import mode

class Language(Enum):
    ENGLISH = 1
    FOREIGN = 2

class LanguagePredictor:

    def __init__(self, nbpass):
        """Init method

        The predictor is not deterministic, and two predictions of the same
        text will produce slightly different results, or it can also predict a
        different language for ambiguous texts. To prevent different prediction,
        an argument "nbpass" allows to specify a number of time we loop the prediction
        method, to maximize the chance that the top language of the text is predicted.

        Args:
            nbpass (int): Number of times the predictor will predict a language
                for a given text in order to choose the one that is predicted
                the most. The larger the number of pass, the most unlikely it is
                that the predictor predicts a language it is not really sure of.
        """

        self.nbpass = nbpass

    def predict(self, X):
        return X.apply(self.detect_str)

    def detect_str(self, string):
        pred = [detect(string) for i in range(self.nbpass)]
        return mode(pred)


