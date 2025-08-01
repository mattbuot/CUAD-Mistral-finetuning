# CUAD-Mistral-finetuning
Fine tuning Mistral models to solve the Contract Understanding Atticus Dataset (CUAD) benchmark

# SETUP

1. setup project using uv
2. download [CUAD](https://www.atticusprojectai.org/cuad) under data/CUAD_v1
3. setup Mistral API key as env variable
4. run generate_dataset.py to create fine-tuning datasets and upload to Mistral
5. use the helpers in utils.py (or the UI) to launch a job
6. run_inference.py to generate predictions
7. evaluate.py to evaluate the results


Tested on MacOS using python 3.12

# RESULTS

| model | task | precision | recall |
|-------|------|-----------|--------|
| baseline | easy | 50% | 51% |
| fine-tuned | easy | 89% | 86% |
| baseline | hard | 16% | 16% |
| fine-tuned | hard | 29% | 29% |

