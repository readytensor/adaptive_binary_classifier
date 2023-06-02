# adaptable_binary_classifier

## Project Description

A adaptable implementation of random forest model created as per Ready Tensor specifications for the binary classification model category.

## Project Structure

```txt
your_chosen_project_name/
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
│   └── unit_tests/
│       ├── (mirrors /src structure)
│       └── ...
├── tmp/
├── .gitignore
├── LICENSE
└── README.md
```

- **`/examples`**: This directory contains all files you want to use as examples. Typically, these would be small data files that can be used as examples for the ML project.
- **`/model_inputs_outputs`**: This directory contains files that are either inputs to, or outputs from, the model. This directory is further divided into:
  - **`/inputs`**: This directory contains all the input files for your project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results. Note that model artifacts should not be saved in this directory. Instead, they should be saved in the `/model/artifacts` directory.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema handler scripts, and `xai` for explainable AI scripts.
- **`/tests`**: This directory contains all the tests for the project. It contains sub-directories for specific types of tests such as unit tests, integration tests, and performance tests. For unit tests, the directory structure should mirror the `/src` directory structure. For example, if you have a `preprocessing` folder in the `/src` directory, then tests corresponding to the scripts under `src/preprocessing` should be placed in the `/tests/unit_tests/preprocessing` directory.
- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains an MIT license for the project. You can change this to any other license you want.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`requirements`**: This directory contains the requirements files. You may create multiple requirements files for different purposes, such as `requirements.txt` for the main code in the `src` directory, `requirements_text.txt` for dependencies required to run tests in the `tests` directory. You may optionally have another requirements file to use for linting and style checks.

## Installation / Setup Instructions

Describe how to get the project setup. Include any dependencies that need to be installed. If it's a Docker container, include instructions for building and running the Docker container.

## Requirements

Specify any system requirements or prerequisite knowledge for using your project. This could include the programming language version, any necessary libraries, or specific hardware requirements.

## Usage Instructions

Describe how to use your project. Include any relevant code snippets or examples.

## Testing

Describe how to run the tests for your project.

## License

Include a section detailing the license your project is under. Do this even if the actual license is in a separate file. This section doesn't need to contain the full text of the license - instead, it can be a brief summary of the license and a reference to the separate LICENSE file.

## Contact Information

Include contact information for the maintainer(s) of the project. This could be an email address, a link to a GitHub profile, or a link to a LinkedIn profile.
