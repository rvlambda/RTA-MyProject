import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    result = model.predict(data)
    #if result == 0:
    #    result = "Slight Injury"
    #elif result == 1:
    #    result = "Serious Injury"
    #else:
    #    result = "Fatal Injury"

    return result
    
