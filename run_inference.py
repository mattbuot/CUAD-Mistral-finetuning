

import os

from mistralai import Any, Mistral

import mistral_utils
import utils
from generate_dataset import extract_cuad_qa_data


def run_inference(client: Mistral, model: str, question_texts: list[str], context_texts: list[str], batch: bool = False) -> list[dict[str, Any]]:
    """Run inference on a set of examples using the Mistral client."""

    examples = []
    for i, (question_text, context_text) in enumerate(zip(question_texts, context_texts)):
        example = utils.generate_conversation_for_question_answering(
            system_prompt=utils.SYSTEM_PROMPT,
            question_text=question_text,
            context_text=context_text
        )
        examples.append(example)

    if batch:
        final_predictions = mistral_utils.batch_inference(client, model, examples)

    else:
        final_predictions = mistral_utils.single_inference(client, model, examples)

    return final_predictions


if __name__ == "__main__":

    dataset_selection = utils.DatasetSelection.TEST    

    qa_data = extract_cuad_qa_data(
        dataset_selection=dataset_selection,
    )

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    model = "ministral-8b-latest"
    model = "ft:ministral-8b-latest:6438ccde:20250730:9394c8fd"
    model = "ft:ministral-8b-latest:6438ccde:20250729:1f13da48"

    predictions = run_inference(client=client, model=model, question_texts=qa_data["questions"], context_texts=qa_data["contracts"], batch=True)
    print()