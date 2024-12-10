import ast

import pandas as pd
from IPython.display import HTML, display


# 문자열이 딕셔너리 형식인지 검사하는 함수
def is_valid_dict_string(row, is_test=False):
    try:
        # 문자열을 딕셔너리로 안전하게 변환 시도
        row_dict = ast.literal_eval(row)

        # 변환된 객체가 원하는 키와 타입을 가진 딕셔너리인지 확인
        return (
            isinstance(row_dict, dict)
            and "question" in row_dict
            and "choices" in row_dict
            and "answer" in row_dict
            and isinstance(row_dict["choices"], list)
            and (isinstance(row_dict["answer"], int) if not is_test else row_dict["answer"] == "")
        )
    except (ValueError, SyntaxError):
        # 변환 실패 시 False 반환
        return False


def str_to_dict(df, column_name, is_test=False):
    # 모든 행이 원하는 딕셔너리 형식인지 확인
    all_valid = df[column_name].apply(lambda row: is_valid_dict_string(row, is_test)).all()
    if all_valid:
        print("모든 행이 올바른 형식입니다. 변환을 진행합니다.")
        # 문자열을 딕셔너리로 변환
        df[column_name] = df[column_name].apply(ast.literal_eval)
    else:
        print("일부 행이 올바른 형식이 아닙니다. 변환을 중단합니다.")


def decompose_problems(df):
    df["question"] = df["problems"].apply(lambda x: x["question"])
    df["choices"] = df["problems"].apply(lambda x: x["choices"])
    df["answer"] = df["problems"].apply(lambda x: x["answer"])


# csv로 저장하고 다시 불러올 때 선택지가 문자열로 저장되기 때문에 리스트로 복원
def str_to_list(string):
    choice_list = ast.literal_eval(string)
    return choice_list


def preprocess_data_columns(data_path, column_name, save_path, is_test=False):
    df = pd.read_csv(data_path)
    str_to_dict(df, column_name, is_test)
    decompose_problems(df)
    df = df.drop(columns=["problems"])
    df["question_plus"] = df["question_plus"].fillna("")
    df.to_csv(save_path, index=False, encoding="utf-8")
    return df


def load_data(data_path):
    df = pd.read_csv(data_path)
    df["question_plus"] = df["question_plus"].fillna("")
    df["choices"] = df["choices"].apply(str_to_list)
    return df


def show_data(index, row):
    style = """
    <style>
        .exam-paper {
            font-family: Arial, sans-serif;
            border: 1px solid #ccc;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .header {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .paragraph {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        .question {
            font-size: 1.1em;
            font-weight: bold;
            margin-top: 15px;
        }
        .choices p {
            margin: 5px 0;
            font-size: 1em;
        }
        .answer {
            color: red;
            text-decoration: underline;
            font-weight: bold;
        }
        .text {
            font-size: 1.2em;
            font-weight: normal;
            line-height: 1.7;
        }
        .explanation {
            font-size: 1.1em;
            color: #333;
            margin-top: 15px;
            line-height: 1.5;
        }
        hr {
            border: 1px solid #ddd;
            margin: 15px 0;
        }
    </style>
    """

    # 시험지 형식 HTML
    exam_html = f"""
    <div class="exam-paper">
        <div class="header">Index: {index} | ID: {row['id']}</div>
        <div class="paragraph">지문:</div>
        <p class="text">{row['paragraph']}</p>
    """

    # 'question_plus'가 있으면 지문과 문제 사이에 보기로 추가
    if "question_plus" in row and pd.notna(row["question_plus"]) and row["question_plus"].strip() != "":
        exam_html += f"""
        <hr>
        <div class="paragraph">보기:</div>
        <p class="text">{row['question_plus']}</p>
        """

    exam_html += f"""
        <hr>
        <div class="question">문제:</div>
        <p class="text">{row['question']}</p>
        <hr>
        <div class="choices">
            <h4>선택지:</h4>
    """

    # 선택지 표시 (정답 인덱스를 기준으로 강조)
    for i, choice in enumerate(row["choices"]):
        if i + 1 == row["answer"]:
            exam_html += f"<p class='answer'>{i + 1}. {choice}</p>"
        else:
            exam_html += f"<p>{i + 1}. {choice}</p>"

    exam_html += """
            </div>
        </div>
    </div>
    """

    # 'llm_response'가 존재하고 값이 결측치가 아니면 해설 추가
    if "llm_response" in row and pd.notna(row["llm_response"]):
        exam_html += f"""
                </div>
                <div class="explanation">
                    <h4>해설:</h4>
                    <p>{row['llm_response']}</p>
                </div>
            </div>
        </div>
        """

    # 스타일 적용하여 시험지 표시
    display(HTML(style + exam_html))


def view_data(df, start_index, end_index):
    for index, row in df.loc[start_index:end_index].iterrows():
        show_data(index=index, row=row)


def labeling(df, start_index, end_index, save_path, label_columns):
    for index, row in df.loc[start_index:end_index].iterrows():
        show_data(index=index, row=row)

        # label_columns 리스트에 대응하는 열로 숫자 입력 받기
        for label_column in label_columns:
            label = input(f"{label_column}: ")
            # Boolean 변환 시도
            # "True", "False"는 대소문자 구분 없이 처리
            if label.lower() == "true":
                label = True
            elif label.lower() == "false":
                label = False

            try:
                label = int(label)
            except ValueError:
                pass
            print(f"{label_column}: {label}")
            df.loc[index, label_column] = label
            df.to_csv(save_path, index=False, encoding="utf-8")
