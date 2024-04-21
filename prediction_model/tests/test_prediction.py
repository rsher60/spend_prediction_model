import pytest
import pandas as pd
from prediction_model.config import config
from prediction_model.processing .data_handling import  load_dataset , load_pipeline
from prediction_model.predict import generate_predictions , generate_predictions_test

"""

classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    return pred
"""



@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    output = generate_predictions_test(single_row)
    result = {"prediction": output}
    return result


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None


def test_single_pred_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],list)

def test_single_pred_validation(single_prediction):
    assert single_prediction.get('prediction')[0] == 1


# output from predict script is not null

# output from predict is int data type
# the output is a particular category when tested with a sample output
