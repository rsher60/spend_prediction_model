import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = 'mushroom_train_1.csv'
TEST_FILE = 'mushroom_test.csv'

MODEL_NAME = 'imb_classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'target_mapped'

#Final features used in the model
FEATURES = ['age', 'gender', 'income', 'education', 'region',
       'loyalty_status', 'purchase_frequency', 'purchase_amount',
       'product_category', 'promotion_usage', 'satisfaction_score']

NUM_FEATURES = ['age','income','satisfaction_score']

CAT_FEATURES = ['gender',
 'education',
  'region',
  'loyalty_status',
 'purchase_frequency','product_category','promotion_usage']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['gender','education','region','loyalty_status',
 'purchase_frequency','product_category','promotion_usage'  ]

FEATURE_TO_MODIFY = ['income']

#DROP_FEATURES = ['target']

LOG_FEATURES = ['income'] # taking log of numerical columns