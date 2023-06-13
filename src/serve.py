from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

from serve_utils import (
    ModelResources,
    combine_predictions_response_with_explanations,
    generate_unique_request_id,
    get_model_resources,
    transform_req_data_and_make_predictions,
)
from xai.explainer import get_explanations_from_explainer


def create_app(model_resources):

    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        """GET endpoint that returns a message indicating the service is running.

        Returns:
            dict: A dictionary with a "message" key and "Pong!" value.
        """
        return {"message": "Pong!"}

    class InferenceRequestBodyModel(BaseModel):
        """
        A Pydantic BaseModel for handling inference requests.

        Attributes:
            instances (list): A list of input data instances.
        """

        instances: List[dict]

    @app.post("/infer", tags=["inference"], response_class=JSONResponse)
    async def infer(request: InferenceRequestBodyModel) -> dict:
        """POST endpoint that takes input data as a JSON object and returns
        predicted class probabilities.

        Args:
            request (InferenceRequestBodyModel): The request body containing the
                input data.

        Returns:
            dict: dict: A dictionary with "status", "message", "timestamp", "requestId",
                "targetClasses", "targetDescription", and "predictions" keys.
        """
        request_id = generate_unique_request_id()
        data = pd.DataFrame.from_records(request.dict()["instances"])
        _, predictions_response = await transform_req_data_and_make_predictions(
            data, model_resources, request_id
        )
        return predictions_response

    @app.post("/explain", tags=["explanations", "XAI"], response_class=JSONResponse)
    async def explain(request: InferenceRequestBodyModel) -> dict:
        """POST endpoint that takes input data as a JSON object and returns
        the predicted class probabilities with explanations.

        Args:
            request (InferenceRequestBodyModel): The request body containing
                the input data.

        Raises:
            HTTPException: If there is an error during inference.

        Returns:
            dict: A dictionary with "status", "message", "timestamp", "requestId",
                "targetClasses", "targetDescription", "predictions",
                and "explanationMethod" keys.
        """
        request_id = generate_unique_request_id()
        (
            transformed_data,
            predictions_response,
        ) = await transform_req_data_and_make_predictions(
            request, model_resources, request_id
        )
        explanations = get_explanations_from_explainer(
            instances_df=transformed_data,
            explainer=model_resources.explainer,
            predictor_model=model_resources.predictor_model,
            class_names=model_resources.data_schema.target_classes,
        )
        predictions_response = combine_predictions_response_with_explanations(
            predictions_response=predictions_response, explanations=explanations
        )
        return predictions_response

    return app


def create_and_run_app(model_resources: ModelResources):
    """Create and run Fastapi app for inference service

    Args:
        model (ModelResources, optional): The model resources instance.
            Defaults to load model resources from paths defined in paths.py.
    """
    app = create_app(model_resources)
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    create_and_run_app(model_resources=get_model_resources())
