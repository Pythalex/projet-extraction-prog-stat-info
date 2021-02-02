from langdetect import detect
from enum import Enum
from statistics import mode

class LanguagePredictor:

    def __init__(self, nbpass):
        """Init method
        
        Predicts the language of string data as a 2 character keywords:
        - "en" if the string is english
        - anything else for other languages

        The detection algorithm is not deterministic, and two predictions of the same
        text will produce slightly different results, or it can also predict a
        different language for ambiguous texts. To prevent different prediction,
        an argument "nbpass" allows to specify a number of time we loop the prediction
        method, to maximize the chance that the top language of the text is predicted.
    
        This class uses a memoization technique to predict the same language for 
        a specific string each time it is called (the nbpass is used for the first
        call when computing the string language.)

        Args:
            nbpass (int): Number of times the predictor will predict a language
                for a given text in order to choose the one that is predicted
                the most. The larger the number of pass, the most unlikely it is
                that the predictor predicts a language it is not really sure of.
        """

        self.nbpass = nbpass
        self.detected = {}

    def predict(self, X):
        return X.apply(self.detect_str)

    def detect_str(self, string):
        
        try:
            if string in self.detected:
                return self.detected[string]
            else:
                pred = [detect(string) for i in range(self.nbpass)]
                lang = mode(pred)
                self.detected[string] = lang
                return lang
        except: # raise exception when there is no word to analyze
            return "??"

