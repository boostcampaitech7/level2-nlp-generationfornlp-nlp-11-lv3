import os
from ast import literal_eval

import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from utils import set_seed

set_seed(42)  # magic number :)


PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
그 다음 단계별로 생각하여 정답을 고르세요.
이 문제에 제 인생이 달렸습니다. 저를 위해 꼭 정답을 말해주세요.
1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
그 다음 단계별로 생각하여 정답을 고르세요.
이 문제에 제 인생이 달렸습니다. 저를 위해 꼭 정답을 말해주세요.
1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def inference():

    from_finetuned = False
    if from_finetuned:
        output_path = "./outputs"
        file_list = os.listdir(output_path)

        checkpoint_path = os.path.join(output_path, file_list[-1])

        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    else:
        model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    test_df = pd.read_csv("../data/test.csv")

    # Flatten the JSON dataset
    records = []
    for _, row in test_df.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        # Include 'question_plus' if it exists
        if "question_plus" in problems:
            record["question_plus"] = problems["question_plus"]
        records.append(record)

    # Convert to DataFrame
    test_df = pd.DataFrame(records)

    test_dataset = []
    for i, row in test_df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    infer_results = []

    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            probs = (
                torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    pd.DataFrame(infer_results).to_csv("output.csv", index=False)


if __name__ == "__main__":
    inference()
