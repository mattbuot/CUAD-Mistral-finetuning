import ast
import json
from collections import defaultdict
from enum import Enum
from typing import Any

SYSTEM_PROMPT = """
        You are a helpful legal assistant that helps finding relevant information in legal documents. You are tasked to highlight the relevant parts of the provided context that answer the question.
        Always answer questions with a list of short passage of the document that contains the answer. If the question cannot be answered with the context, respond with an empty list.
        If there are multiple relevant passages, return all of them. If the same passage appears multiple times written exactly the same way, return it only once.
        Your output should be an instance of a JSON object following this schema: {"highlighted": [highlighted_part_1, highlighted_part_2, ...]}

        # Example 1:

        Context: "This Marketing Affiliate Agreement (the “Agreement”) is entered into this 8th day of May
            2014, by and between BIRCH FIRST GLOBAL INVESTMENTS INC., a corporation incorporated
            in the U.S. Virgin Islands, with its main place of business located 9100 Havensight, Port of Sale, Ste.
            15/16, St. Thomas, VI 0080 (referred to as “Company”) and MOUNT KNOWLEDGE HOLDINGS
            INC. and/or assigns, a corporation incorporated in the State of Nevada, with its main place of business
            located at 228 Park Avenue S. #56101 New York, NY 10003­1502 (referred to as “Marketing
            Affiliate” or “MA”)."
        Question: "Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"
        Answer: {"highlighted": ['BIRCH FIRST GLOBAL INVESTMENTS INC.', 'MOUNT KNOWLEDGE HOLDINGS INC.']}

        # Example 2:

        Context: same as above
        Question: "When the Eiffel Tower was built?"
        Answer: {"highlighted": []}
    """


class DatasetSelection(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


def generate_conversation_for_question_answering(
    system_prompt: str,
    question_text: str,
    context_text: str,
    answer_text: str | None = None,
) -> list[dict[str, str]]:
    """Generates a conversation standardized for question answering tasks."""
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})

    conversation.append({"role": "user", "content": "Context: " + context_text})
    conversation.append({"role": "user", "content": "Question: " + question_text})

    if answer_text is not None:
        conversation.append(
            {"role": "assistant", "content": json.dumps({"highlighted": answer_text})}
        )
    return conversation


def format_conversations_to_fine_tuning_lines(
    conversations: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Format multiple conversations into a list of lines for fine-tuning."""

    def _conversation_format(
        i: int, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        return {
            "prompt": conversation[1]["content"],
            "prompt_id": str(i),
            "messages": conversation,
        }

    lines = []

    for i, conversation in enumerate(conversations):
        lines.append(_conversation_format(i, conversation))
    return lines


def format_conversations_to_inference_lines(
    conversations: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Format conversations into a list of lines for inference."""

    def _conversation_format(i, conversation):
        return {
            "custom_id": str(i),
            "body": {
                "messages": conversation,
                "response_format": {
                    "type": "json_object",
                },
            },
        }

    lines = []

    for i, conversation in enumerate(conversations):
        lines.append(_conversation_format(i, conversation))
    return lines


def list_to_jsonl(data: list[Any]) -> str:
    """Convert a list of items to a JSONL string."""
    return "\n".join(json.dumps(item) for item in data)


def parse_batch_results(file_path: str, n_queries: int) -> list[list[str]]:
    """Parse the batch results from a JSONL file, several cases are handled in case the output is malformed.

    Importantly, if a prediction is missing because it errored out in the batch, it will be replaced with an empty list."""
    id_to_final_predictions = defaultdict(list)
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            custom_id = int(data["custom_id"])
            response_content = data["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            try:
                predictions = json.loads(response_content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from response content: {response_content}")
                predictions = {"highlighted": []}
            if isinstance(predictions, dict) and "highlighted" in predictions:
                if isinstance(predictions["highlighted"], str):
                    try:
                        predictions = ast.literal_eval(predictions["highlighted"])
                    except SyntaxError:
                        print(
                            f"Error parsing highlighted predictions: {predictions['highlighted']}"
                        )
                        predictions = predictions["highlighted"]
                else:
                    predictions = predictions["highlighted"]
            elif isinstance(predictions, list):
                assert len(predictions) == 0 or isinstance(predictions[0], str), (
                    "Expected predictions to be a list of strings"
                )

            id_to_final_predictions[custom_id] = predictions

    final_results = []
    for i in range(n_queries):
        final_results.append(id_to_final_predictions[i])

    return final_results
