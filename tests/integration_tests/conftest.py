import pytest


@pytest.fixture
def sample_request_data():
    # Define a fixture for test data
    return {
        "instances": [
            {
                "PassengerId": "879",
                "Pclass": 3,
                "Name": "Laleff, Mr. Kristo",
                "Sex": "male",
                "Age": None,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "349217",
                "Fare": 7.8958,
                "Cabin": None,
                "Embarked": "S",
            }
        ]
    }


@pytest.fixture
def sample_response_data():
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": ["0", "1"],
        "targetDescription": "A binary variable indicating whether or \
            not the passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [0.97548, 0.02452],
            }
        ],
    }


@pytest.fixture
def sample_explanation_response_data():
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "2023-05-22T10:51:45.860800",
        "requestId": "0ed3d0b76d",
        "targetClasses": ["0", "1"],
        "targetDescription": "A binary variable indicating whether or not the \
            passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [0.92107, 0.07893],
                "explanation": {
                    "baseline": [0.57775, 0.42225],
                    "featureScores": {
                        "Age_na": [0.05389, -0.05389],
                        "Age": [0.02582, -0.02582],
                        "SibSp": [-0.00469, 0.00469],
                        "Parch": [0.00706, -0.00706],
                        "Fare": [0.05561, -0.05561],
                        "Embarked_S": [0.01582, -0.01582],
                        "Embarked_C": [0.00393, -0.00393],
                        "Embarked_Q": [0.00657, -0.00657],
                        "Pclass_3": [0.0179, -0.0179],
                        "Pclass_1": [0.02394, -0.02394],
                        "Sex_male": [0.13747, -0.13747],
                    },
                },
            }
        ],
        "explanationMethod": "Shap",
    }
