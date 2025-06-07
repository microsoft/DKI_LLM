import os
import re
from typing import Any, Callable, List, Tuple, Optional
import json
import random
import string
from math import isclose
from collections import namedtuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def load_all_examples(file_dir: str) -> dict:
    # data_dir下有多个文件夹，每个文件夹下有多个json文件，每个json文件中包含一个问题
    examples_dic = {}
    for folder in os.listdir(file_dir):
        examples = []
        folder_path = os.path.join(file_dir, folder)
        if os.path.isdir(folder_path):
            files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith('.json')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            for file in files:
                file_path = os.path.join(folder_path, file)
                # print(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    examples.append(json.load(f))
            examples_dic[folder] = examples
    return examples_dic


def extract_model_answer(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, text, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = re.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""


def score_math(expected_output: str, prediction: str) -> Tuple[int]:
    expected_answer = extract_model_answer(expected_output)
    predicted_answer = extract_model_answer(prediction)
    # print(f"expected_answer: {expected_answer}, predicted_answer: {predicted_answer}")
    if math_equal(predicted_answer, expected_answer):
        return True
    else:
        return False
    

def math_equal(prediction: Any, reference: Any) -> bool:
    if str(prediction) == str(reference):
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            return isclose(prediction, reference, abs_tol=1e-3)
    except:
        pass

    try:
        return symbolic_equal(prediction, reference)
    except:
        pass

    return False


def is_digit(num):
    return parse_digits(num) is not None

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except:
        pass
    return False


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def cal_acc(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # # Convert data to a numpy array for easier manipulation
    # data = np.array(data)

    # # List to store the means of bootstrap samples
    # bootstrap_means = []

    # # Generate bootstrap samples and compute the mean for each sample
    # for _ in range(num_bootstrap_samples):
    #     # Resample with replacement
    #     bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    #     # Compute the mean of the bootstrap sample
    #     bootstrap_mean = np.mean(bootstrap_sample)
    #     bootstrap_means.append(bootstrap_mean)

    # # Convert bootstrap_means to a numpy array for percentile calculation
    # bootstrap_means = np.array(bootstrap_means)

    # # Compute the lower and upper percentiles for the confidence interval
    # lower_percentile = (1.0 - confidence_level) / 2.0
    # upper_percentile = 1.0 - lower_percentile
    # ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    # ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # # Compute the median of the bootstrap means
    # median = np.median(bootstrap_means)

    # # Convert to percentages and format to one decimal place
    # ci_lower_percent = ci_lower * 100
    # ci_upper_percent = ci_upper * 100
    # median_percent = median * 100

    # # Return the formatted string with confidence interval and median
    # return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
    # 计算准确率

    data = np.array(data)
    accuracy = np.mean(data)
    return f"Accuracy: {accuracy:.1%}"


def extract_accuracy(fitness_str):
    """
    从字符串中提取 Median 值（float 类型）。

    参数:
        fitness_str (str): 例如 '... Median: 68.6%'

    返回:
        float 或 None: 提取到的 Median 数值（如 68.6），未匹配则返回 None
    """
    match = re.search(r'Accuracy:\s*([\d.]+)%', fitness_str)
    if match:
        return float(match.group(1))
    return None
