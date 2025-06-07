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
from collections import Counter

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def load_all_examples(file_dir: str) -> dict:
    examples_dic = {}
    for file in os.listdir(file_dir):
        with open(os.path.join(file_dir, file), 'r') as f:
            examples_dic[file.split('.')[0]] = json.load(f)
    return examples_dic


def calculate_score(target: str, prediction: str) -> Tuple[float, str]:
    prediction_tokens = normalize_answer(prediction).split()
    targets = target.split("|")
    f1_list = []
    for t in targets:
        ground_truth_tokens = normalize_answer(t).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1_list.append(0)
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_list.append(f1)
    return max(f1_list)

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


# def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
#     """
#     Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
#     Also returns the median of the bootstrap means.
    
#     Args:
#     - data (list or array of float): 1D list or array of data points.
#     - num_bootstrap_samples (int): Number of bootstrap samples.
#     - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
#     Returns:
#     - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
#     """
#     # Convert data to a numpy array for easier manipulation
#     data = np.array(data)

#     # List to store the means of bootstrap samples
#     bootstrap_means = []

#     # Generate bootstrap samples and compute the mean for each sample
#     for _ in range(num_bootstrap_samples):
#         # Resample with replacement
#         bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
#         # Compute the mean of the bootstrap sample
#         bootstrap_mean = np.mean(bootstrap_sample)
#         bootstrap_means.append(bootstrap_mean)

#     # Convert bootstrap_means to a numpy array for percentile calculation
#     bootstrap_means = np.array(bootstrap_means)

#     # Compute the lower and upper percentiles for the confidence interval
#     lower_percentile = (1.0 - confidence_level) / 2.0
#     upper_percentile = 1.0 - lower_percentile
#     ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
#     ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

#     # Compute the median of the bootstrap means
#     median = np.median(bootstrap_means)

#     # Convert to percentages and format to one decimal place
#     ci_lower_percent = ci_lower * 100
#     ci_upper_percent = ci_upper * 100
#     median_percent = median * 100

#     # Return the formatted string with confidence interval and median
#     return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"


# def extract_median(fitness_str):
#     """
#     从字符串中提取 Median 值（float 类型）。

#     参数:
#         fitness_str (str): 例如 '... Median: 68.6%'

#     返回:
#         float 或 None: 提取到的 Median 数值（如 68.6），未匹配则返回 None
#     """
#     match = re.search(r'Median:\s*([\d.]+)%', fitness_str)
#     if match:
#         return float(match.group(1))
#     return None

def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
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
    return f"F1 Score: {accuracy:.1%}"


def extract_median(fitness_str):
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