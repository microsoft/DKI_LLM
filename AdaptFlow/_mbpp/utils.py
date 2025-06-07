import os
import re
from typing import Any, Callable, List, Tuple, Optional, Dict
import json
import random
import string
from math import isclose
from collections import namedtuple
import threading
import time

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from sanitize import sanitize
import numpy as np
from collections import Counter

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def load_all_examples(file_dir: str) -> dict:
    examples_dic = {}
    for file in os.listdir(file_dir):
        with open(os.path.join(file_dir, file), 'r') as f:
            examples_dic[file.split('.')[0]] = json.load(f)
    return examples_dic

def run_with_timeout(func, timeout):
    result = []
    stop_event = threading.Event()

    def target():
        try:
            result.append(func())
        except Exception as e:
            result.append(e)
        finally:
            stop_event.set()

    thread = threading.Thread(target=target)
    thread.start()
    is_timeout = not stop_event.wait(timeout)

    if is_timeout:
        raise TimeoutError("Function execution timed out")

    if not result:
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]

def check_solution(solution, test, entry_point):
    solution = sanitize(code=solution, entrypoint=entry_point)
    try:
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": __import__("re"),
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Optional": Optional,
            "Any": Any,
        }

        exec(solution, global_dict)

        if entry_point not in global_dict:
            raise ValueError(f"Function {entry_point} is not defined in the solution.")

        exec(test, global_dict)

        check = global_dict["check"]

        result = run_with_timeout(check, 15)

        if result is None:
            result = ("PASS", "The solution passed all test cases.")

    except TimeoutError:
            result = (
                "FAIL",
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
    except Exception as e:
        # print(f"Error in check_solution: {e}")
        error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
        result = ("FAIL", error_message)

        with open("error.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

    return result


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
    return f"{accuracy:.1%}"

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

    data = np.array(data)
    accuracy = np.mean(data)
    return f"Accuracy: {accuracy:.1%}"


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