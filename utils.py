import json
from typing import Any

SYSTEM_PROMPT = """
        You are a helpful legal assistant that helps finding relevant information in legal documents. You are tasked to highlight the relevant parts of the provided context that answer the question.
        Always answer questions with a list of short passage of the document that contains the answer. If the question cannot be answered with the context, respond with an empty list.
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


def format_conversations_to_fine_tuning_jsonl(
    conversations: list[list[dict[str, str]]],
) -> str:
    """Format multiple conversations into a JSONL string."""
    jsonl_lines = []

    for i, conversation in enumerate(conversations):
        jsonl_lines.append(json.dumps(_conversation_format(i, conversation)))
    return "\n".join(jsonl_lines)


def _conversation_format(i: int, conversation: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "prompt": conversation[1]["content"],
        "prompt_id": str(i),
        "messages": conversation,
    }
