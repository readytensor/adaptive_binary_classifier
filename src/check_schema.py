from config import paths
from schema.data_schema import load_json_data_schema, save_schema


def read_schema():
    """Reads the binary classification schema."""

    # load the json file schema into a dictionary and use it to instantiate
    # the schema provider
    data_schema = load_json_data_schema(paths.INPUT_SCHEMA_DIR)

    # check if schema provider parsed the schema correctly
    print(f"id is: `{data_schema.id}`")
    print(f"target is: `{data_schema.target}`")
    print(f"allowed values for target are: `{data_schema.target_classes}`")
    print(f"positive_class is: `{data_schema.positive_class}`")
    print(f"Numerical features are: {data_schema.numeric_features}")
    print(f"Categorical features are: {data_schema.categorical_features}")
    print(
        "Allowed values for Categorical features are:"
        f"{data_schema.allowed_categorical_values}"
    )
    print(f"All fields are: {data_schema.all_fields}")

    # save schema to model artifacts directory
    save_schema(schema=data_schema, output_path=paths.SAVED_SCHEMA_PATH)


if __name__ == "__main__":
    read_schema()
