from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import numpy as np

classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation',pp.ModeImputer(variables=config.CAT_FEATURES)),

        ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
       # ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)