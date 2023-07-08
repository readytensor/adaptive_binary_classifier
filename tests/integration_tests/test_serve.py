import pytest
from fastapi.testclient import TestClient

from src.serve import create_app
from src.train import run_training
from src.serve_utils import get_model_resources


@pytest.fixture
def app(
        input_schema_dir,
        train_dir,
        config_file_paths_dict: dict,
        resources_paths_dict: dict,
    ):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """
    # extract paths to all config files
    model_config_file_path = config_file_paths_dict["model_config_file_path"]
    preprocessing_config_file_path = \
        config_file_paths_dict["preprocessing_config_file_path"]
    default_hyperparameters_file_path = \
        config_file_paths_dict["default_hyperparameters_file_path"]
    hpt_specs_file_path = config_file_paths_dict["hpt_specs_file_path"]
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]

    # Create temporary paths for all outputs/artifacts
    saved_schema_path = resources_paths_dict["saved_schema_path"]
    pipeline_file_path = resources_paths_dict["pipeline_file_path"]
    target_encoder_file_path = resources_paths_dict["target_encoder_file_path"]
    predictor_file_path = resources_paths_dict["predictor_file_path"]
    hpt_results_file_path = resources_paths_dict["hpt_results_file_path"]
    explainer_file_path = resources_paths_dict["explainer_file_path"]

    # Run the training process without hyperparameter tuning
    run_tuning = False
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        preprocessing_config_file_path=preprocessing_config_file_path,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path,
        run_tuning=run_tuning,
        hpt_specs_file_path=hpt_specs_file_path if run_tuning else None,
        hpt_results_file_path=hpt_results_file_path if run_tuning else None,
        explainer_config_file_path=explainer_config_file_path,
        explainer_file_path=explainer_file_path,
    )

    # create model resources dictionary
    model_resources = get_model_resources(**resources_paths_dict)

    # create test app
    return TestClient(create_app(model_resources))


def test_infer_endpoint_integration(
        app,
        sample_request_data,
        sample_response_data,
        schema_provider
    ):
    """
    End-to-end integration test for the /infer endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the
    /infer endpoint, and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the
    test_model_resources fixture, and the application's dependency on ModelResources
    is overridden to use this instance for the test.

    The function sends a POST request to the "/infer" endpoint with the
    test_sample_request_data using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and
    compares specific values in the returned response_data with the
    sample_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        app (TestClient): The test client for the FastAPI application.
        sample_request_data (dict): The fixture for test request data.
        sample_response_data (dict): The fixture for expected response data.
        schema_provider (Any): The fixture for the schema provider.
    Returns:
        None
    """
    response = app.post("/infer", json=sample_request_data)
    response_data = response.json()

    # assertions
    assert set(response_data.keys()) == set(response.json().keys())
    assert (
        response_data["predictions"][0]["sampleId"]
        == sample_response_data["predictions"][0]["sampleId"]
    )
    assert (
        response_data["predictions"][0]["predictedClass"]
        in schema_provider.target_classes
    )


def test_explain_endpoint_integration(
    app,
    sample_request_data,
    sample_explanation_response_data,
    schema_provider,
):
    """
    End-to-end integration test for the /explain endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the
    /explain endpoint, and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the
    test_model_resources fixture, and the application's dependency on ModelResources
    is overridden to use this instance for the test.

    The function sends a POST request to the "/explain" endpoint with the
    test_request_data using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys,
    and compares specific values in the explanation_response_data with the
    sample_explanation_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        app (TestClient): The test client for the FastAPI application.
        sample_request_data (dict): The fixture for test request data.
        sample_explanation_response_data (dict): The fixture for expected explanation
            response data.
        schema_provider (Any): The fixture for the schema provider.
    Returns:
        None
    """
    response = app.post("/explain", json=sample_request_data)
    explanation_response_data = response.json()

    # assertions
    assert set(explanation_response_data.keys()) == set(
        sample_explanation_response_data.keys()
    )
    assert (
        explanation_response_data["predictions"][0]["sampleId"]
        == sample_explanation_response_data["predictions"][0]["sampleId"]
    )
    assert (
        explanation_response_data["predictions"][0]["predictedClass"]
        in schema_provider.target_classes
    )

    # baseline assertions
    assert (
        explanation_response_data["predictions"][0]["explanation"].get("baseline")
        is not None
    )
    baseline = explanation_response_data["predictions"][0]["explanation"]["baseline"]
    assert len(baseline) == 2
    assert round(sum(baseline), 4) == 1.0000

    # explanation assertions
    # feature scores are not None
    assert (
        explanation_response_data["predictions"][0]["explanation"].get("featureScores")
        is not None
    )
    #verify numeric features are in feature scores
    feature_scores = explanation_response_data["predictions"][0]["explanation"][
        "featureScores"
    ]
    features_in_scores = feature_scores.keys()
    assert all(
        feature in features_in_scores
        for feature in schema_provider.numeric_features
    )
