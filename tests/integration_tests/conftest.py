import pytest

from src.xai.explainer import ShapClassificationExplainer


@pytest.fixture
def sample_request_data(schema_dict):
    # Define a fixture for test request data
    sample_dict = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        if feature["dataType"]=="NUMERIC":
            sample_dict[feature["name"]] = feature["example"]
        elif feature["dataType"]=="CATEGORICAL":
            sample_dict[feature["name"]] = feature["categories"][0]
    return {"instances": [{**sample_dict}]}


@pytest.fixture
def sample_response_data(schema_dict):
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": schema_dict["target"]["classes"],
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                # unknown because we don't know the predicted class
                "predictedClass": "unknown",
                # predicted probabilities are made up for this test
                "predictedProbabilities": [0.5, 0.5],
            }
        ],
    }


@pytest.fixture
def sample_explanation_response_data(schema_dict):
    # Define a fixture for expected response
    feature_scores = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        feature_scores[feature["name"]] =[42, 42]
    return {
        "status": "success",
        "message": "",
        # made up timestamp
        "timestamp": "2023-05-22T10:51:45.860800",
        "requestId": "made_up_id",
        "targetClasses": schema_dict["target"]["classes"],
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                # unknown because we don't know the predicted class
                "predictedClass": "unknown",
                # predicted probabilities are made up for this test
                "predictedProbabilities": [0.5, 0.5],
                "explanation": {
                    # all values are made up
                    "baseline": [42, 42],
                    "featureScores": {**feature_scores},
                },
            }
        ],
        "explanationMethod": ShapClassificationExplainer.EXPLANATION_METHOD,
    }
