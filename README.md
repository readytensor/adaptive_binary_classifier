# Adaptable Binary Classifier Implementation

## Project Description

This repository is a dockerized implementation of the random forest binary classifier. It is implemented in flexible way that it can be used with any binary classification dataset with with use of a data schema. The model includes:

- A flexible preprocessing pipeline built using **SciKit-Learn** and **feature-engine**
- A Random Forest algorithm built using **SciKit-Learn**
- Hyperparameter-tuning using **scikit-optimize**
- SHAP explainer using the **shap** package
- **FASTAPI** inference service which provides endpoints for predictions and local explanations.
- **Pydantic** data validation is used for the schema, training and test files, as well as the inference request data.
- Error handling and logging using **Python's logging** module.
- Comprehensive set of unit, integration, coverage and performance tests using **pytest**, **pytest-cov**.

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users. The purpose of the tutorial series is to help AI developers create adaptable algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

The tutorial series is divided into 3 modules as follows:

1. **Model development**: This section will cover the following tutorials:

   - **Creating Standardized Project Structure**: This tutorial will review the standardized project structure we will use in this series.
   - **Using Data Schemas**: This tutorial discusses how data schemas can be used to avoid hard-coding implementations to specific datasets.
   - **Creating Adaptable Data Preprocessing-Pipelines**: In this tutorial, we cover how to create data preprocessing pipelines that can work with diverse set of algorithms such as tree-based models and neural networks.
   - **Building a Binary Classifier in Python**: This tutorial reviews a binary classifier model implementation which enables providing a common interface for various types of algorithms.
   - **Hyper-Parameter Tuning (HPT) Using SciKit-Optimize**: This tutorial covers how to use SciKit-Optimize to perform hyper-parameter tuning on ML models.
   - **Integrating FastAPI for Online Inference**: This tutorial covers how we can set up an inference service using FastAPI to provide online predictions from our machine learning model.
   - **Model Interpretability with Shapley Values**: This tutorial describes implementing an eXplainable AI (XAI) technique called Shap to provide interpretability to ML models.

2. **Quality Assurance for ML models**: This section will cover the following tutorials:

   - **Error Handling and Logging**: In this tutorial, we review how to add basic error handling and logging to ML model implementations.
   - **Data Validation Using Pydantic**: This tutorial covers how we can use the pydantic library to validate input data for our machine learning model implementation.
   - **Testing ML Model Implementations**: This tutorial covers the topic of testing machine learning model implementations including unit, integration, coverage and performance testing. We will also introduce tox for environment management and test automation.

3. **Model Containerization for Deployment**: This section will cover the following tutorials:

   - **Containerizing ML Models - The Model-as-a-Service pattern**: In this tutorial, we review how to containerize an ML model to make it easily portable and deployable in different environments. We will use the Model-as-a-Service pattern for deployment.
   - **Containerizing ML Models - The Hybrid Pattern**: In this tutorial, we cover how to containerize an ML model with Docker to include both training and inference services.

This particular branch called [module-1-complete](https://github.com/readytensor/adaptable_binary_classifier/tree/module-1-complete) is the completion point of module 1 in the series. The module 1 start point is [here](https://github.com/readytensor/adaptable_binary_classifier/tree/module-1-start).

## Project Structure

```txt
adaptable_binary_classifier/
├── examples/
├── model_inputs_outputs/
│   ├── inputs/
│   │   ├── data/
│   │   │   ├── testing/
│   │   │   └── training/
│   │   └── schema/
│   ├── model/
│   │   └── artifacts/
│   └── outputs/
│       ├── errors/
│       ├── hpt_outputs/
│       └── predictions/
├── requirements/
│   ├── requirements.txt
│   └── requirements_text.txt
├── src/
│   ├── config/
│   ├── data_models/
│   ├── hyperparameter_tuning/
│   ├── prediction/
│   ├── preprocessing/
│   ├── schema/
│   └── xai/
├── tests/
│   ├── integration_tests/
│   ├── performance_tests/
│   ├── test_resources/
│   ├── test_results/
│   │   ├── coverage_tests/
│   │   └── performance_tests/
│   └── unit_tests/
│       ├── (mirrors /src structure)
│       └── ...
├── tmp/
├── .gitignore
├── LICENSE
└── README.md
```

- **`/examples`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`/model_inputs_outputs`**: This directory contains files that are either inputs to, or outputs from, the model. This directory is further divided into:

  - **`/inputs`**: This directory contains all the input files for this project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.

  - **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.

  - **`/outputs`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results.

- **`requirements`**: This directory contains the requirements files. We have multiple requirements files for different purposes:

  - `requirements.txt` for the main code in the `src` directory

  - `requirements_text.txt` for dependencies required to run tests in the `tests` directory.

  - `requirements_quality.txt` for dependencies related to formatting and style checks.

- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories :

  - **`config`**: for configuration files such as hyperparameters, model configuration, hyperparameter tuning-configuration specs, paths, and preprocessing configuration.

  - **`data_models`**: for data models for input validation including the schema, training and test files, and the inference request data.

  - **`hyperparameter_tuning`**: for hyperparameter-tuning (HPT) functionality using scikit-optimize

  - **`prediction`**: for prediction model (random forest) script

  - **`preprocessing`**: for data preprocessing scripts including the preprocessing pipeline, target encoder, and custom transformers.

  - **`schema`**: for schema handler script.

  - **`xai`**: explainable AI scripts (SHAP explainer)

  Furthermore, we have the following scripts in `src`:

  - **`check_preprocessing.py`**: This is a temporary script created to check the preprocessing pipeline. It will be removed in module 2 of this series.

  - **`check_schema.py`**: This is also a temporary script created to check the schema. It will be removed in module 2 of this series.

  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.

  - **`serve.py`**: This script is used to serve the model as a REST API. It loads the artifacts and creates a FastAPI server to serve the model. It provides 3 endpoints: `/ping`, `/infer`, and `/explain`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions. The `/explain` endpoint is used to get local explanations for the predictions.

  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.

  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`. It also saves a SHAP explainer object in the path `./model/artifacts/`. When the train task is run with a flag to perform hyperparameter tuning, it also saves the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.

  - **`utils.py`**: This script contains utility functions used by the other scripts.

- **`/tests`**: This directory contains all the tests for the project and associated resources and results.

  - **`integration_tests.py`**: This directory contains the integration tests for the project. We cover four main workflows: data preprocessing, training, prediction, and inference service.

  - **`performance_tests.py`**: This directory contains performance tests for the training and batch prediction workflows in the script `test_train_predict.py`. It also contains performance tests for the inference service workflow in the script `test_inference_apis.py`. Helper functions are defined in the script `performance_test_helpers.py`. Fixtures and other setup are contained in the script `conftest.py`.

  - **`test_resources.py`**: This folder contains various resources needed in the tests, such as trained model artifacts (including the preprocessing pipeline, target encoder, explainer, etc.). These resources are used in integration tests and performance tests.

  - **`test_results.py`**: This folder contains the results for the performance tests. These are persisted to disk for later analysis.

  - **`unit_tests.py`**: This folder contains all the unit tests for the project. It is further divided into subdirectories mirroring the structure of the `src` folder. Each subdirectory contains unit tests for the corresponding script in the `src` folder.

- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.

- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.

- **`LICENSE`**: This file contains the license for the project.

- **`pytest.ini`**: This is the configuration file for pytest, the testing framework used in this project.

- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.

- **`tox.ini`**: This is the configuration file for tox, the primary test runner used in this project.

## Usage

To run the project:

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively.
- Run the script `src/train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints. The service runs on port 8080.

## Testing

### Running through Tox

To run the tests:
Tox is used for running the tests. To run the tests, simply run the command:

```bash
tox
```

This will run the tests as well as formatters `black` and `isort` and linter `flake8`. You can run tests corresponding to specific environment, or specific markers. Please check `tox.ini` file for configuration details.

### Running through Pytest

- Run the command `pytest` from the root directory of the repository.
- To run specific scripts, use the command `pytest <path_to_script>`.
- To run slow-running tests (which take longer to run): use the command `pytest -m slow`.

## Requirements

Dependencies are listed in the file `requirements.txt`. These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```

For testing, dependencies are listed in the file `requirements-test.txt`. You can install these packages by running the following command:

```python
pip install -r requirements-test.txt
```

Alternatively, you can let tox handle the installation of test dependencies for you. To do this, simply run the command `tox` from the root directory of the repository.

## Contact Information

Repository created by Ready Tensor, Inc.
