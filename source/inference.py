import os
from ast import literal_eval

# import evaluate
import hydra
import numpy as np
import pandas as pd
import torch

# import transformers
from datasets import Dataset

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from omegaconf import DictConfig
from peft import AutoPeftModelForCausalLM  # LoraConfig

# from rag import init_vectorstore, retrieve_query
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer  # BitsAndBytesConfig

# from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from utils import set_seed

home_path = os.path.expanduser("~")

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
그 다음 단계별로 생각하여 정답을 고르세요.
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
1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def logit_inference(cfg: DictConfig):

    model_id = cfg.model
    seed = cfg.seed
    data_path = cfg.data_path
    output_path = cfg.output_path
    model_path = cfg.model_path
    from_finetuned = cfg.inference.from_fine_tuning

    set_seed(seed)  # magic number :)
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    output_path = os.path.join(os.path.dirname(__file__), output_path)
    data_path = os.path.join(os.path.dirname(__file__), data_path)
    if from_finetuned:
        file_list = os.listdir(output_path)

        checkpoint_path = os.path.join(model_path, file_list[-1])

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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    data_path = os.path.join(data_path, "test.csv")
    test_df = pd.read_csv(data_path)

    # Flatten the JSON dataset
    records = []
    for _, row in tqdm(test_df.iterrows()):
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

    # embeddings_model = HuggingFaceEmbeddings(
    #     model_name="jhgan/ko-sbert-nli",
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )
    # vectorstore_path = "/db/vectorstore"
    # vectorstore_path = os.path.join(os.path.dirname(__file__), vectorstore_path)

    # if os.path.exists(vectorstore_path):
    #     print("Loading vectorstore")
    #     vectorstore = FAISS.load_local(vectorstore_path, embeddings_model, allow_dangerous_deserialization=True)
    # else:
    #     vectorstore = init_vectorstore(vectorstore_path)

    # Convert to DataFrame
    test_df = pd.DataFrame(records)

    test_dataset = []
    for i, row in tqdm(test_df.iterrows(), desc="Processing data and retrieving documents"):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])
        # total_text = row["paragraph"] + row["question"]
        # if len(total_text) < 300:
        #     doc = retrieve_query(total_text, vectorstore)
        #     row["paragraph"] = row["paragraph"] + " 힌트: " + doc[0].page_content
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
                    {
                        "role": "system",
                        "content": "당신은 학생에게 문제를 풀어주는 선생님입니다. 질문에 대한 답을 구하세요.",
                    },
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
    model_name = model_id.replace("/", "_")
    output = f"output_train_{model_name}.csv"
    output_path = os.path.join(output_path, output)
    pd.DataFrame(infer_results).to_csv(output_path, index=False)


def generate_inference(cfg: DictConfig):

    model_id = cfg.model
    seed = cfg.seed
    data_path = cfg.data_path
    output_path = cfg.output_path
    # from_finetuned = cfg.inference.from_fine_tuning

    set_seed(seed)  # magic number :)
    output_path = os.path.join(os.path.dirname(__file__), output_path)
    data_path = os.path.join(os.path.dirname(__file__), data_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    data_path = os.path.join(data_path, "test.csv")
    dataset = pd.read_csv(data_path)

    # Flatten the JSON dataset
    records = []
    for _, row in dataset.iterrows():
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
    df = pd.DataFrame(records)
    df["question_plus"] = df["question_plus"].fillna("")
    df["full_question"] = df.apply(
        lambda x: x["question"] + " " + x["question_plus"] if x["question_plus"] else x["question"], axis=1
    )

    # Calculate the length of each question
    df["question_length"] = df["full_question"].apply(len)

    dataset = Dataset.from_pandas(df)
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])
        len_choices = len(dataset[i]["choices"])
        # <보기>가 있을 때
        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

        # chat message 형식으로 변환
        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 학생에게 문제를 풀어주는 선생님입니다. 질문에 대한 답을 구하세요.",
                    },
                    {"role": "user", "content": user_message},
                ],
                "label": dataset[i]["answer"],
                "len_choices": len_choices,
            }
        )
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # 데이터 토큰화
    tokenized_dataset = processed_dataset.map(
        tokenize,
        remove_columns=list(processed_dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    infer_results = []

    # pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    model.eval()
    with torch.inference_mode():
        for i, data in enumerate(tqdm(tokenized_dataset, desc="Inferencing")):
            id = processed_dataset[i]["id"]
            text = tokenizer.decode(data["input_ids"], skip_special_tokens=False)
            input_text = text.split("<end_of_turn>")[0]

            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_text = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :],  # noqa: E203
                skip_special_tokens=True,
            )[0]
            generated_text = generated_text.strip()
            infer_results.append({"id": id, "answer": generated_text})
    model_name = model_id.replace("/", "_")
    output = f"output_train_{model_name}.csv"
    output_path = os.path.join(output_path, output)
    pd.DataFrame(infer_results).to_csv(output_path, index=False)


@hydra.main(config_path="../configs", config_name="configs")
def main(cfg: DictConfig):
    if cfg.inference.inference_mode == "logit":
        print("logit inference")
        logit_inference(cfg)
    elif cfg.inference.inference_mode == "generate":
        print("generate inference")
        generate_inference(cfg)


if __name__ == "__main__":
    print(os.getcwd())
    main()
