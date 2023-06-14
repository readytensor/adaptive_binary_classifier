import os
import shutil
import time
from typing import List

import docker
import pytest
import requests
from docker.errors import ContainerError

client = docker.from_env()


@pytest.fixture
def script_dir() -> str:
    """
    Returns the directory of the current script.

    Returns:
        str: Path to the directory of the current script.
    """
    return os.path.dirname(os.path.abspath(__file__))


def move_files_to_temp_dir(src_dir: str, dst_dir: str, file_list: List[str]) -> None:
    """
    Copies specified files from the source directory to the destination directory.

    Args:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
        file_list (List[str]): List of filenames to be copied.
    """
    for file in file_list:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        assert os.listdir(src_dir)


@pytest.fixture
def mounted_volume_for_training(tmpdir: str, test_resources_path: str) -> str:
    """
    Prepares and returns a directory with a specific structure and files for the
    training task.

    Args:
        tmpdir (str): Pytest fixture for creating temporary directories.
        test_resources_path (str): Path to the test resources directory.

    Returns:
        str: Path to the prepared directory.
    """
    base_dir = tmpdir.mkdir("model_inputs_outputs")
    # Create the necessary directories
    input_dir = base_dir.mkdir("inputs")
    input_data_dir = input_dir.mkdir("data")
    training_dir = input_data_dir.mkdir("training")
    testing_dir = input_data_dir.mkdir("testing")
    schema_dir = input_dir.mkdir("schema")
    base_dir.mkdir("model").mkdir("artifacts")
    output_dir = base_dir.mkdir("outputs")
    output_dir.mkdir("errors")
    output_dir.mkdir("hpt_outputs")
    output_dir.mkdir("predictions")

    # Move the necessary files to the created directories
    move_files_to_temp_dir(
        test_resources_path, str(training_dir), ["titanic_train.csv"]
    )
    move_files_to_temp_dir(test_resources_path, str(testing_dir), ["titanic_test.csv"])
    move_files_to_temp_dir(
        test_resources_path, str(schema_dir), ["titanic_schema.json"]
    )

    return str(base_dir)


@pytest.fixture
def mounted_volume_for_inference(tmpdir: str, test_resources_path: str) -> str:
    """
    Prepares and returns a directory with a specific structure and files
    for the inference task.

    Args:
        tmpdir (str): Pytest fixture for creating temporary directories.
        test_resources_path (str): Path to the test resources directory.

    Returns:
        str: Path to the prepared directory.
    """
    base_dir = tmpdir.mkdir("model_inputs_outputs")
    # Create the necessary directories
    inputs_dir = base_dir.mkdir("inputs")
    inputs_dir.mkdir("data").mkdir("testing")
    inputs_dir.mkdir("schema")
    model_dir = base_dir.mkdir("model").mkdir("artifacts")
    outputs_dir = base_dir.mkdir("outputs")
    outputs_dir.mkdir("errors")
    outputs_dir.mkdir("hpt_outputs")
    outputs_dir.mkdir("predictions")

    # Move the necessary files to the created directories
    move_files_to_temp_dir(
        test_resources_path,
        str(base_dir.join("inputs/data/testing")),
        ["titanic_test.csv"],
    )
    move_files_to_temp_dir(
        test_resources_path,
        str(base_dir.join("inputs/schema")),
        ["titanic_schema.json"],
    )

    # Move the model artifacts to the created directory
    move_files_to_temp_dir(
        test_resources_path,
        str(model_dir),
        [
            "explainer.joblib",
            "pipeline.joblib",
            "predictor.joblib",
            "schema.joblib",
            "target_encoder.joblib",
        ],
    )

    return str(base_dir)


@pytest.fixture
def image_name():
    """Fixture that returns the name of the Docker image to be used in testing.

    Returns:
        str: Docker image name for testing.
    """
    return "test-image"


@pytest.fixture
def docker_image(script_dir: str, image_name: str):
    """Fixture to build and remove docker image."""
    # Build the Docker image
    dockerfile_path = os.path.join(script_dir, "../../")
    client.images.build(path=dockerfile_path, tag=image_name)
    yield image_name

    # Remove the Docker image
    client.images.remove(image_name)


@pytest.fixture
def container_name():
    """Fixture that returns the name of the Docker container to be used in testing.

    Returns:
        str: Docker container name for testing.
    """
    return "test-container"


@pytest.mark.slow
def test_training_task(
    mounted_volume_for_training: str, docker_image: str, container_name: str
):
    """
    Integration test for the training task.

    Args:
        mounted_volume_for_training (str): The path of the training data directory.
        docker_image (str): The name of the Docker image.
        container_name (str): The name of the Docker container.

    Raises:
        exc: If the Docker container exits with an error.
    """
    volumes = {
        mounted_volume_for_training: {"bind": "/opt/model_inputs_outputs", "mode": "rw"}
    }
    try:
        _ = client.containers.run(
            docker_image,
            "train",
            name=container_name,
            volumes=volumes,
            remove=True,
        )
    except ContainerError as exc:
        print(f"Container exited with error. Exit status: {exc.exit_status}")
        print(f"Standard error: {exc.stderr}")
        raise exc  # Re-raise the exception to fail the test case

    model_path = os.path.join(mounted_volume_for_training, "model/artifacts/")
    assert os.listdir(model_path)  # Assert that the directory is not empty


@pytest.mark.slow
def test_prediction_task(
    mounted_volume_for_inference: str, docker_image: str, container_name: str
):
    """
    Integration test for the prediction task.

    Args:
        mounted_volume_for_inference (str): The path of the inference data directory.
        docker_image (str): The name of the Docker image.
        container_name (str): The name of the Docker container.

    Raises:
        exc: If the Docker container exits with an error.
    """
    volumes = {
        mounted_volume_for_inference: {
            "bind": "/opt/model_inputs_outputs",
            "mode": "rw",
        }
    }
    try:
        _ = client.containers.run(
            docker_image,
            "predict",
            name=container_name,
            volumes=volumes,
            remove=True,
        )
    except ContainerError as exc:
        print(f"Container exited with error. Exit status: {exc.exit_status}")
        print(f"Standard error: {exc.stderr}")
        raise exc  # Re-raise the exception to fail the test case

    prediction_path = os.path.join(mounted_volume_for_inference, "outputs/predictions/")
    assert os.listdir(prediction_path)  # Assert that the directory is not empty


@pytest.mark.slow
def test_inference_service(
    mounted_volume_for_inference: str,
    docker_image: str,
    container_name: str,
    sample_request_data: dict,
    sample_response_data: dict,
    sample_explanation_response_data: dict,
):
    """
    Integration test for the inference service.

    Args:
        mounted_volume_for_inference (str): The path of the inference data directory.
        docker_image (str): The name of the Docker image.
        container_name (str): The name of the Docker container.
        sample_request_data (dict): The sample request data for testing the `/infer`
            and `/explain` endpoints.
        sample_response_data (dict): The expected response data for testing the
            `/infer` endpoint.
        sample_explanation_response_data (dict): The expected response data for testing
            the `/explain` endpoint.

    Raises:
        exc: If the Docker container exits with an error.
    """
    volumes = {
        mounted_volume_for_inference: {
            "bind": "/opt/model_inputs_outputs",
            "mode": "rw",
        }
    }

    container = client.containers.create(
        docker_image,
        command="serve",
        name=container_name,
        volumes=volumes,
        ports={"8080/tcp": 8080},
    )

    try:
        container.start()

        # Wait for the service to start.
        time.sleep(5)

        # Test `/ping` endpoint
        response = requests.get("http://localhost:8080/ping", timeout=5)
        assert response.status_code == 200

        # Test `/infer` endpoint
        response = requests.post(
            "http://localhost:8080/infer", json=sample_request_data, timeout=5
        )
        response_data = response.json()
        assert response.status_code == 200
        assert response_data["targetClasses"] == sample_response_data["targetClasses"]
        print(response_data["targetDescription"])
        print(sample_response_data["targetDescription"])
        assert (
            response_data["predictions"][0]["predictedClass"]
            == sample_response_data["predictions"][0]["predictedClass"]
        )

        # Test `/explain` endpoint
        response = requests.post(
            "http://localhost:8080/explain", json=sample_request_data, timeout=5
        )
        response_data = response.json()
        assert response.status_code == 200
        assert (
            response_data["targetClasses"]
            == sample_explanation_response_data["targetClasses"]
        )
        assert (
            response_data["predictions"][0]["predictedClass"]
            == sample_explanation_response_data["predictions"][0]["predictedClass"]
        )
        # explanations
        assert "explanation" in response_data["predictions"][0]
        assert "baseline" in response_data["predictions"][0]["explanation"]
        assert "featureScores" in response_data["predictions"][0]["explanation"]

    except ContainerError as exc:
        print(f"Container exited with error. Exit status: {exc.exit_status}")
        print(f"Standard error: {exc.stderr}")
        raise exc  # Re-raise the exception to fail the test case

    finally:
        container.stop()
        container.remove()