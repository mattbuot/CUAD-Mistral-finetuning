import json
import os

from mistralai import Mistral

import utils
from mistral_utils import upload_dataset

CATEGORIES_WITH_AT_LEAST_10_PERCENT_LABELS = [
    '"Document Name"',
    '"Parties"',
    '"Agreement Date"',
    '"Effective Date"',
    '"Expiration Date"',
    '"License Grant"',
    '"Exclusivity"',
    '"Renewal Term"',
    '"Non-Transferable License"',
    '"Minimum Commitment"',
    '"Revenue/Profit Sharing"',
]


def extract_cuad_qa_data(
    dataset_selection: utils.DatasetSelection,
    include_empty_labels: bool | None = True,
    contract_character_limit: int = 10_000,
) -> dict[str, list[str]]:
    """Extracts question-answer pairs from the CUAD dataset with the associated contract."""

    if dataset_selection == utils.DatasetSelection.TRAIN:
        contract_start_index = 0
        contract_end_index = 400
    elif dataset_selection == utils.DatasetSelection.TEST:
        contract_start_index = 400
        contract_end_index = 510
    else:
        assert dataset_selection == utils.DatasetSelection.VALIDATION, (
            f"Invalid dataset selection {dataset_selection}"
        )
        contract_start_index = 400
        contract_end_index = 450

    with open("data/CUAD_v1/CUAD_v1.json") as json_file:
        data = json.load(json_file)

    contracts = []
    questions = []
    labels = []

    for contract_number, contract_info in enumerate(data["data"]):
        if (
            contract_number >= contract_start_index
            and contract_number < contract_end_index
        ):
            print(f"Processing contract {contract_number + 1}/{len(data['data'])}")
            for paragraph_number, paragraph_infos in enumerate(
                contract_info["paragraphs"]
            ):
                print(
                    f"Processing paragraph {paragraph_number + 1}/{len(contract_info['paragraphs'])} in contract {contract_number + 1}"
                )
                for i, q in enumerate(paragraph_infos["qas"]):
                    question = paragraph_infos["qas"][i]["question"]
                    contract = paragraph_infos["context"]

                    contract_chunk = contract[:contract_character_limit]
                    label_list = [
                        answer["text"]
                        for answer in paragraph_infos["qas"][i]["answers"]
                        if answer["text"] in contract_chunk
                    ]

                    if any(
                        topic in question
                        for topic in CATEGORIES_WITH_AT_LEAST_10_PERCENT_LABELS
                    ):
                        include_empty_labels_condition = (
                            include_empty_labels and len(label_list) == 0
                        )
                        include_non_empty_labels_condition = (
                            not include_empty_labels and len(label_list) > 0
                        )
                        label_condition = (
                            include_empty_labels is None
                            or include_empty_labels_condition
                            or include_non_empty_labels_condition
                        )
                        if label_condition:
                            labels.append(label_list)
                            questions.append(question)
                            contracts.append(contract_chunk)

    return {"contracts": contracts, "questions": questions, "labels": labels}


def dump_fine_tuning_dataset(
    questions: list[str], contexts: list[str], answers: list[str], file_name: str
) -> None:
    """Generates a fine-tuning dataset in JSONL format for question answering tasks and stores it locally."""

    print(f"Generating fine-tuning dataset in {file_name}...")
    
    examples = []

    for i, (question_text, context_text, answer) in enumerate(
        zip(questions, contexts, answers)
    ):
        example = utils.generate_conversation_for_question_answering(
            system_prompt=utils.SYSTEM_PROMPT,
            question_text=question_text,
            context_text=context_text,
            answer_text=answer,
        )
        examples.append(example)

    lines_data = utils.format_conversations_to_fine_tuning_lines(examples)
    jsonl_data = utils.list_to_jsonl(lines_data)
    with open(file_name, "w") as f:
        f.write(jsonl_data)

    print(f"Fine-tuning dataset saved to {file_name}")


if __name__ == "__main__":
    dataset_selection = utils.DatasetSelection.TRAIN

    qa_data = extract_cuad_qa_data(
        dataset_selection=dataset_selection,
        include_empty_labels=False,
    )

    file_name = f"data/cuad_{dataset_selection.value}_fine_tuning_dataset.jsonl"

    dump_fine_tuning_dataset(
        questions=qa_data["questions"],
        contexts=qa_data["contracts"],
        answers=qa_data["labels"],
        file_name=file_name,
    )

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    upload_dataset(client=client, file_name=file_name)
