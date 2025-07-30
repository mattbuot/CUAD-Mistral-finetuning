import json
from time import sleep

from mistralai import Mistral
from pydantic import BaseModel

import utils


class Highlights(BaseModel):
    highlighted: list[str]


def upload_dataset(client: Mistral, file_name: str) -> None:
    """Uploads a dataset stored locally to Mistral."""

    print("Uploading dataset to Mistral...")
    dataset = client.files.upload(
        file={
            "file_name": file_name,
            "content": open(file_name, "rb"),
        }
    )
    print(f"Dataset uploaded with ID: {dataset.id}")


def create_fine_tuning_job(
    client: Mistral,
    training_files: list[dict[str, str]],
    validation_files: list[str],
    model: str,
    hyperparameters: dict,
) -> None:
    """Creates a fine-tuning job on Mistral using the uploaded dataset."""

    print("Creating fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        model=model,
        training_files=training_files,
        validation_files=validation_files,
        hyperparameters=hyperparameters,
        invalid_sample_skip_percentage=0.1,
        auto_start=False,
    )
    print(f"Fine-tuning job created with ID: {job.id}")


def single_inference(
    client: Mistral, model: str, examples: list[dict[str, str]]
) -> list[list[str]]:
    """Perform inference on a single example using the Mistral client."""
    final_predictions = []

    for i, example in enumerate(examples):
        print(f"Processing question {i + 1}/{len(examples)}")

        response = client.chat.parse(
            model=model,
            messages=example,
            response_format=Highlights,
        )
        final_predictions.append(
            json.loads(response.choices[0].message.content)["highlighted"]
        )
    return final_predictions


def batch_inference(
    client: Mistral, model: str, examples: list[list[dict[str, str]]]
) -> list[list[str]]:
    """Run inference in batch mode."""

    lines_data = utils.format_conversations_to_inference_lines(examples)
    jsonl_data = utils.list_to_jsonl(lines_data)
    with open("data/batch_input.jsonl", "w") as f:
        f.write(jsonl_data)

    final_predictions = []
    batch_data = client.files.upload(
        file={
            "file_name": "data/batch_input.jsonl",
            "content": open("data/batch_input.jsonl", "rb"),
        },
        purpose="batch",
    )

    created_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"},
    )

    status = "INITIAL"
    while status not in ["SUCCESS", "FAILED"]:
        retrieved_job = client.batch.jobs.get(job_id=created_job.id)
        status = retrieved_job.status
        print(f"Awaiting for job to complete, status: {status}")
        sleep(10)

    if status == "FAILED":
        raise RuntimeError(f"Batch job failed with error: {retrieved_job.error}")

    output_file_stream = client.files.download(file_id=retrieved_job.output_file)

    with open("data/batch_results.jsonl", "wb") as f:
        f.write(output_file_stream.read())

    final_predictions = utils.parse_batch_results(
        "data/batch_results.jsonl", n_queries=len(examples)
    )
    return final_predictions
