import os

from mistralai import Mistral

import mistral_utils
import utils
from generate_dataset import extract_cuad_qa_data


def run_inference(
    client: Mistral,
    model: str,
    question_texts: list[str],
    context_texts: list[str],
    batch: bool = False,
) -> list[list[str]]:
    """Run inference on a set of examples using the Mistral client."""

    examples = []
    for i, (question_text, context_text) in enumerate(
        zip(question_texts, context_texts)
    ):
        example = utils.generate_conversation_for_question_answering(
            system_prompt=utils.SYSTEM_PROMPT,
            question_text=question_text,
            context_text=context_text,
        )
        examples.append(example)

    if batch:
        final_predictions = mistral_utils.batch_inference(client, model, examples)

    else:
        final_predictions = mistral_utils.single_inference(client, model, examples)

    return final_predictions


def store_predictions_and_labels(
    predictions: list[list[str]], labels: list[list[str]], file_name: str
) -> None:
    """Store predictions and labels in a JSONL file."""
    data = []
    for prediction, label in zip(predictions, labels):
        data.append({"prediction": prediction, "label": label})

    jsonl_data = utils.list_to_jsonl(data)
    with open(file_name, "w") as f:
        f.write(jsonl_data)

    print(f"Predictions and labels saved to {file_name}")


if __name__ == "__main__":
    dataset_selection = utils.DatasetSelection.TEST

    qa_data = extract_cuad_qa_data(
        dataset_selection=dataset_selection,
    )

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    model = "ministral-8b-latest"
    model = "ft:ministral-8b-latest:6438ccde:20250730:9394c8fd"

    predictions = run_inference(
        client=client,
        model=model,
        question_texts=qa_data["questions"],
        context_texts=qa_data["contracts"],
        batch=True,
    )
    store_predictions_and_labels(
        predictions=predictions,
        labels=qa_data["labels"],
        file_name=f"data/cuad_{dataset_selection.value}_predictions.jsonl",
    )
