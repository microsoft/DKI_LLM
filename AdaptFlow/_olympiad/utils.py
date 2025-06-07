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
import re
import json
import sympy as sp
from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex
import sys
import math

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


SPECIAL_SIGNAL_MAP = {
    "\\left": "",
    "\\right": "",
    "∶": ":",
    "，": ",",
    "$": "",
    "\\approx": "=",
    "\\simeq": "=",
    "\\sim": "=",
    "^\\prime": "'",
    "^{\\prime}": "'",
    "^\\circ": "",
    "%": "",
}
PRECISION = 1e-4
PI = parse_latex("\\pi")

def judge(expression1, expression2, precision=1e-8):
    # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
    # Default precision is a list for supporting multiple expressions
    precision = precision if isinstance(precision, list) else [precision]

    try:
        expression1, expression2 = preprocess(expression1, expression2)
    except:
        return False
    if expression1 == expression2:
        # print("Exactly equal")
        return True
    
    # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
    expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
    expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)
    
    expression1 = split_by_comma(expression1)
    expression2 = split_by_comma(expression2)

    temp_list1 = trans_plus_minus_sign(expression1)
    temp_list2 = trans_plus_minus_sign(expression2)

    # Set up a list for allowed errors
    if len(precision) <= 1:
        precision = precision * len(temp_list1)
    
    if len(temp_list1) != len(temp_list2):
        return False

    # Check if elements in both lists can be paired and are equal
    idx = -1
    while len(temp_list1) != 0:
        idx = (idx + 1) % len(temp_list1)

        item1 = temp_list1[idx]
        PRECISION = precision[idx]

        for item2 in temp_list2:
            if is_equal(item1, item2):
                temp_list1.remove(item1)
                temp_list2.remove(item2)
                precision.remove(PRECISION)
                break
        else:
            # If no match was found, return False
            return False

    # If all elements are matched, return True
    return True

def preprocess(expression1, expression2):
    # Preprocess expressions to extract and replace special symbols
    def extract_boxed_content(latex_str):
        boxed_matches = re.finditer(r'\\boxed{', latex_str)
        results = ""

        for match in boxed_matches:
            start_index = match.end()
            end_index = start_index
            stack = 1

            while stack > 0 and end_index < len(latex_str):
                if latex_str[end_index] == '{':
                    stack += 1
                elif latex_str[end_index] == '}':
                    stack -= 1
                end_index += 1

            if stack == 0:
                content = latex_str[start_index:end_index - 1]
                results += content + ","
            else:
                raise ValueError("Mismatched braces in LaTeX string.")

        if results == "":
            last_line_ans = latex_str.strip().split("\n")[-1]
            dollar_pattern = r"\$(.*?)\$"
            answers = re.findall(dollar_pattern, last_line_ans)

            if answers:
                for ans in answers:
                    results += ans + ","
            else:
                results = latex_str
            
        return results
    
    def sepcial_symbol_replace(expression):
        if "\\in " in expression:
            expression = expression.split("\\in ")[1]
        
        for signal in SPECIAL_SIGNAL_MAP:
            expression = expression.replace(signal, SPECIAL_SIGNAL_MAP[signal])

        expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

        pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
        expression = re.sub(pattern, r'\1', expression)

        return expression
    
    exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
    exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

    return exp1, exp2

def split_by_comma(expr: str):
    # Splits expressions by commas outside of brackets
    in_bracket_num = 0
    splitted_expr = []
    start_idx = 0
    for i, char in enumerate(expr):
        if char in ["(", "["]:
            in_bracket_num += 1
        elif char in [")", "]"]:
            in_bracket_num -= 1
        elif char == "," and in_bracket_num == 0:
            splitted_expr.append(expr[start_idx:i].strip())
            start_idx = i + 1

    if start_idx < len(expr):
        splitted_expr.append(expr[start_idx:].strip())   
    
    return splitted_expr

def trans_plus_minus_sign(expr_list: list):
    # Translates plus-minus signs into separate expressions
    new_expr_list = []
    for expr in expr_list:
        if "\\pm" in expr:
            new_expr_list.append(expr.replace("\\pm", "+"))
            new_expr_list.append(expr.replace("\\pm", "-"))
        else:
            new_expr_list.append(expr)
    
    return new_expr_list

def is_equal(expression1, expression2):
    # Default first expression is ground truth. Check if expressions are equal in different aspects
    if expression1 == expression2 and expression1 != "" and expression2 != "":
        # print("Equivalent natively")
        return True

    # First check if both are intervals
    if is_interval(expression1) and is_interval(expression2):
        try:
            if interval_equal(expression1, expression2):
                # print("Interval equivalent")
                return True
        except:
            return False

    # Then check for numerical equality
    try:
        if numerical_equal(expression1, expression2):
            # print("Numerically equivalent")
            return True
    except:
        pass
    
    # Then check if expressions are mathematically equal
    try:
        if expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
            # print("Expression equivalent")
            return True
    except:
        pass
        
    # Lastly, check for equation equality
    try:
        if equation_equal(expression1, expression2):
            # print("Equation equivalent")
            return True
    except:
        pass
        
    return False

def is_interval(expr):
    # Checks if an expression is an interval
    return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

def interval_equal(expression1, expression2):
    # Check if two intervals are mathematically equivalent
    def compare_two_interval(inter1, inter2):
        if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
            return False
        
        inter1 = inter1.strip('[]()')
        inter2 = inter2.strip('[]()')

        items_1 = inter1.split(',')
        items_2 = inter2.split(',')

        for item_1, item_2 in zip(items_1, items_2):
            if not expression_equal(item_1, item_2):
                return False
        return True
        
    interval1 = expression1
    interval2 = expression2

    if interval1 == interval2:
        return True
    else:
        inter_list1 = interval1.split("\\cup")
        inter_list2 = interval2.split("\\cup")
        
        if len(inter_list1) != len(inter_list2):
            return False
        else:
            for inter1, inter2 in zip(inter_list1, inter_list2):
                if not compare_two_interval(inter1, inter2):
                    return False
            return True

def numerical_equal(expression1: str, expression2: str, include_percentage: bool = True):
    # Check if two numerical values are equal within an allowed error range
    # Includes possible percentage cases
    reference = float(expression1)
    prediction = float(expression2)
    
    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    
    for item in gt_result:
        if abs(item - prediction) <= PRECISION * 1.01:
            return True
    return False

def expression_equal(exp1, exp2):
    # Check if two expressions are mathematically equivalent
    # Extract expression and use sympy for equivalence checking
    def extract_expression(expression):
        if "=" in expression:
            expression = expression.split("=")[1]
        return expression.strip()
    
    exp1 = extract_expression(exp1)
    exp2 = extract_expression(exp2)

    expr1_sym = sympify(parse_latex(exp1))
    expr2_sym = sympify(parse_latex(exp2))

    if expr1_sym == expr2_sym:
        return True
    else:
        expr1_sym = sympy_sub_pi(expr1_sym)
        expr2_sym = sympy_sub_pi(expr2_sym)

        if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
            return False
        elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
            try:
                if not (can_compute_power(expr1_sym) and can_compute_power(expr2_sym)):
                    print(f"These two numbers cannot be calculated by the current computer for: \"{str(expr1_sym)}\" and \"{str(expr2_sym)}\"")
                    return False

                if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= PRECISION * 1.01:
                    return True
                else:
                    return False
            except:
                return False
        else:
            try:
                simplified_expr = simplify(expr1_sym - expr2_sym)

                num_value = simplified_expr.evalf()
                
                return abs(num_value) < 1e-3
            except:
                return False
            
def equation_equal(expression1, expression2):
    # Check if two equations are mathematically equivalent
    # Simplify equations and use sympy for equivalence checking
    def simplify_equation(latex_eq):
        lhs, rhs = latex_eq.split('=')

        lhs_expr = parse_latex(lhs)
        rhs_expr = parse_latex(rhs)

        equation = Eq(lhs_expr, rhs_expr)

        simplified_eq = simplify(equation.lhs - equation.rhs)

        return simplified_eq

    expr1_sym = simplify_equation(expression1)
    expr2_sym = simplify_equation(expression2)

    division_result_1 = simplify(expr1_sym / expr2_sym)
    division_result_2 = simplify(expr2_sym / expr1_sym)

    if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
        return True
    else:
        return False

def sympy_sub_pi(expression_sympy):
    # Replaces the symbol for pi in sympy expressions with its numerical value
    return expression_sympy.subs(PI, math.pi)

def can_compute_power(expr):
    # Checks if a power expression can be computed
    if isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        if base.is_number and exp.is_number:
            MAX_EXP = 1000  # Adjust based on computing environment
            if abs(exp.evalf()) > MAX_EXP:
                return False
            else:
                return True
        else:
            return False
    else:
        return True  # Not a power expression, can compute
def load_all_examples(file_dir: str) -> dict:
    examples_dic = {}
    for file in os.listdir(file_dir):
        with open(os.path.join(file_dir, file), 'r') as f:
            examples_dic[file.split('.')[0]] = json.load(f)
    return examples_dic


def is_correct(prediction, target):
    try:
        num1 = float(prediction)
        num2 = float(target)
        return num1 == num2
    except Exception as e:
        return False


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