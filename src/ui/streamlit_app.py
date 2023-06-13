import streamlit as st
import pandas as pd

from serve_utils import (
    get_model_resources,
    generate_unique_request_id,
    transform_req_data_and_make_predictions
)


@st.cache_data()
def get_cached_model_resources():
    return get_model_resources()


def render_form_and_get_input(model_resources):
    st.title("Prediction App")
    st.markdown("""---""")
    schema = model_resources.data_schema
    st.header(f"Dataset: {schema.title}")
    # with st.form("123"):
    st.subheader("Input Sample for Prediction")
    input_data = {}
    # add id
    input_data[schema.id] = st.text_input(f'Enter {schema.id}')
    # Loop through each feature in the schema
    for feature in schema.features:
        description = schema.get_description_for_feature(feature)
        is_nullable = schema.is_feature_nullable(feature)

        if feature in schema.numeric_features:
            # Create numeric input field
            example_value = schema.get_example_value_for_feature(feature)
            input_data[feature] = st.number_input(description, value=example_value)
        elif feature in schema.categorical_features:
            # Create select box for categorical features
            options = schema.get_allowed_values_for_categorical_feature(feature)
            input_data[feature] = st.selectbox(description, options)

        # Add checkbox if feature is nullable
        if is_nullable:
            input_data[feature] = None \
                if st.checkbox(f"Set {description} as Null?", value=False) \
                    else input_data[feature]
    return input_data

def make_prediction(input_data, model_resources):
    # Generate a unique request id
    request_id = generate_unique_request_id()
    data = pd.DataFrame.from_records([input_data])
    print(data)
    _, predictions = \
        transform_req_data_and_make_predictions(
        data, model_resources, request_id)
    print(predictions)
    return predictions

def render_prediction(predictions, model_resources):
    with st.container():
        st.markdown("""---""")
        schema = model_resources.data_schema
        target_classes = model_resources.data_schema.target_classes
        sample_prediction = predictions['predictions'][0]
        st.subheader("Prediction")
        st.markdown(f"__Predicted class: `{sample_prediction['predictedClass']}`__")
        st.write(f"Target description: {schema.target_description}")
        st.write(
            f"Probability of class {target_classes[0]}: "
                f"`{sample_prediction['predictedProbabilities'][0]}`")
        st.write(
            f"Probability of class {target_classes[1]}: "
                f"`{sample_prediction['predictedProbabilities'][1]}`")
        st.markdown("""---""")


def run_ui_app():
    model_resources = get_cached_model_resources()
    input_data = render_form_and_get_input(model_resources)
    # Predict button
    if st.button('Predict'):
        prediction = make_prediction(input_data, model_resources)
        render_prediction(prediction, model_resources)
