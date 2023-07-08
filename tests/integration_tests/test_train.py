import os

import pytest

from src.train import run_training


@pytest.mark.slow
@pytest.mark.parametrize("run_tuning", [False, True])
def test_run_training(
    run_tuning: bool,
    input_schema_dir: str,
    train_dir: str,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
) -> None:
    """Test the run_training function to make sure it produces the required artifacts.

    This test function checks whether the run_training function runs end-to-end
    without errors and produces the expected artifacts. It does this by running
    the training process with and without hyperparameter tuning. After each run,
    it verifies that the expected artifacts have been saved to disk at the correct
    paths.

    Args:
        run_tuning (bool): Boolean indicating whether to run hyperparameter
            tuning or not.
        input_schema_dir (str): Path to the input schema directory.
        model_config_file_path (str): Path to the model configuration file.
        train_dir (str): Path to the training directory.
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

    # Run the training process without tuning
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

    # Assert that the model artifacts are saved in the correct paths
    assert os.path.isfile(saved_schema_path)
    assert os.path.isfile(pipeline_file_path)
    assert os.path.isfile(target_encoder_file_path)
    assert os.path.isfile(predictor_file_path)
    assert os.path.isfile(explainer_file_path)
    if run_tuning:
        assert os.path.isfile(hpt_results_file_path)
