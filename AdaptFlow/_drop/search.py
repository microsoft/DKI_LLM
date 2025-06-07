import argparse
import copy
import json
import os
import random
import requests
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
import backoff
from tqdm import tqdm

from drop_prompt import get_init_archive, get_prompt, get_meta_prompt, get_adaptation_prompt, get_reflexion_prompt, get_fail_reflexion_prompt

from utils import load_all_examples, calculate_score, random_id, bootstrap_confidence_interval, extract_median

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

client = openai.OpenAI(
    api_key="",
    base_url=""
)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=8192,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(msg_list, model, temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=8192,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-4o-mini-20240718', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = args.model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Answer the question using only the correct entity name or phrase. Do not add any explanation, reasoning, or full sentence. Respond with the answer only." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    key_list = list(load_all_examples(args.train_dir_path).keys())
    key_list.append("all")
    dir_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive_{args.version}")
    archive = {}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    inner_start_dict = {}
    for key in key_list:
        file_path = os.path.join(dir_path, f"{key}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                archive[key] = json.load(json_file)
            if key == "all":
                if "generation" in archive[key][-1] and "label" in archive[key][-1] and archive[key][-1]['label'] == "outer_loop" and isinstance(archive[key][-1]['generation'], int):
                    out_start = archive[key][-1]['generation']
                else:
                    out_start = 0
            else:
                if "label" in archive[key][-1] and archive[key][-1]['label'] == "outer_loop":
                    inner_start_dict[key] = archive[key][-2]['generation']
                elif "label" in archive[key][-1] and archive[key][-1]['label'] == "inner_loop":
                    inner_start_dict[key] = archive[key][-1]['generation']
                else:
                    inner_start_dict[key] = 0
        else:
            archive[key] = copy.deepcopy(get_init_archive())
            inner_start_dict[key] = 0
            out_start = 0

    acc_list_dic = {}
    for key in key_list:
        acc_list_dic[key] = {}
        if key == "all":
            continue
        for solution in archive[key]:
            if 'fitness' in solution:
                continue

            solution['generation'] = "initial"
            print(f"============Initial Archive {key}: {solution['name']}=================")
            try:
                acc_list, _ = evaluate_forward_fn_key(args, solution["code"], key, args.train_dir_path)
                acc_list_dic[key][solution['name']] = acc_list
            except Exception as e:
                print("During evaluating initial archive:")
                print(e)
                continue

            fitness_str = bootstrap_confidence_interval(acc_list)
            solution['fitness'] = fitness_str
            
            # save results
            file_path = os.path.join(dir_path, f"{key}.json")
            with open(file_path, 'w') as json_file:
                json.dump(archive[key], json_file, indent=4)
    
    # calculate the fitness when key == all
    for solution in archive["all"]:
        if 'fitness' in solution:
            continue
        acc_list = []
        for key in key_list:
            if key == "all":
                continue
            acc_list.extend(acc_list_dic[key][solution['name']])
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['generation'] = "initial"
        solution['fitness'] = fitness_str

        # save results
        file_path = os.path.join(dir_path, "all.json")
        with open(file_path, 'w') as json_file:
            json.dump(archive["all"], json_file, indent=4)
        
    # outer loop
    for n_outer in range(out_start, args.n_outer_loop):
        print(f"============Outer Loop {n_outer + 1}=================")
        
        # 使用ThreadPoolExecutor并行处理不同的key
        with ThreadPoolExecutor(max_workers=(len(key_list))) as executor:
            future_to_key = {
                executor.submit(
                    process_key_inner_loop, 
                    args, key, archive[key], dir_path, 
                    n_outer, out_start, inner_start_dict
                ): key for key in key_list if key != "all"
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    archive[key] = future.result()
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    continue

        # outer meta
        system_prompt, prompt = get_meta_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.update_model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n_outer -= 1
            continue
        
        acc_list_dic = {}
        for key in key_list:
            acc_list_dic[key] = {}

        # 使用ThreadPoolExecutor并行处理不同的key
        with ThreadPoolExecutor(max_workers=(len(key_list))) as executor:
            future_to_key = {
                executor.submit(
                    process_key_outer_evaluation, 
                    key, 
                    n_outer,
                    copy.deepcopy(next_solution),
                    dir_path,
                    archive,
                    acc_list_dic,
                    False
                ): key for key in key_list if key != "all"
            }
            
            acc_list = []
            fail_examples_list = []
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    if result is not None:
                        acc_list.extend(result[0])
                        fail_examples_list.extend(result[1])
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    continue

        solution = copy.deepcopy(next_solution)
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str
        solution['generation'] = n_outer + 1
        solution['label'] = "outer_loop"

        archive["all"].append(solution)

        # save results
        file_path = os.path.join(dir_path, "all.json")
        with open(file_path, 'w') as json_file:
            json.dump(archive["all"], json_file, indent=4)

        # 利用失败case进行reflection
        # ________________________________________________________________________________________________________
        if len(fail_examples_list) > 0:
            reflexion_prompt = get_fail_reflexion_prompt(random.sample(fail_examples_list, 5))
            msg_list.append({"role": "assistant", "content": str(solution)})
            msg_list.append({"role": "user", "content": reflexion_prompt})
        
        next_solution = get_json_response_from_gpt_reflect(msg_list, args.update_model)
        solution = copy.deepcopy(next_solution)
        acc_list_dic = {}
        for key in key_list:
            acc_list_dic[key] = {}

        # 使用ThreadPoolExecutor并行处理不同的key
        with ThreadPoolExecutor(max_workers=(len(key_list))) as executor:
            future_to_key = {
                executor.submit(
                    process_key_outer_evaluation, 
                    key, 
                    n_outer,
                    copy.deepcopy(next_solution),
                    dir_path,
                    archive,
                    acc_list_dic,
                    True
                ): key for key in key_list if key != "all"
            }
            
            acc_list = []
            fail_examples_list = []
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    if result is not None:
                        acc_list.extend(result[0])
                        fail_examples_list.extend(result[1])
                except Exception as e:
                    print(f"Error processing key {key}: {e}")
                    continue

        solution = copy.deepcopy(next_solution)
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str
        solution['generation'] = n_outer + 1
        solution['label'] = "outer_loop"
        solution['special_label'] = "fail_reflexion"

        archive["all"].append(solution)

        # save results
        file_path = os.path.join(dir_path, "all.json")
        with open(file_path, 'w') as json_file:
            json.dump(archive["all"], json_file, indent=4)
        # ________________________________________________________________________________________________________

        
def evaluate_forward_fn_key(args, forward_str, key, file_dir):
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    examples_dic = load_all_examples(file_dir)
    random.seed(args.shuffle_seed)


    examples = examples_dic[key]
    random.shuffle(examples)

    questions = [example['context'] for example in examples]
    answers = [example['ref_text'] for example in examples]
    print(f"{key} problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1
    task_queue = []
    for p in questions:
        taskInfo = Info('task', 'User', p, -1)
        task_queue.append(taskInfo)
    agentSystem = AgentSystem()
    f1_list = []
    fail_examples_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))
    for idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            if type(extracted_answer) == int:
                extracted_answer = str(extracted_answer)
            correct_answer = answers[idx]
            f1 = calculate_score(correct_answer, extracted_answer)
        except Exception as e:
            f1_list.append(0)
            # 添加字符串，要包含信息problem, solution, extracted_answer
            fail_examples_list.append(f"problem: {questions[idx]}, correct_answer: {answers[idx]}, extracted_answer: {extracted_answer}")
            continue
        f1_list.append(f1)
        if not f1:
            fail_examples_list.append(f"problem: {questions[idx]}, correct_answer: {answers[idx]}, extracted_answer: {extracted_answer}")
    print(f"acc: {bootstrap_confidence_interval(f1_list)}")
    return f1_list, fail_examples_list


def process_key_inner_loop(args, key, archive, dir_path, n_outer, out_start, inner_start_dict):
    if key == "all":
        return
    
    if n_outer == out_start:
        inner_start = inner_start_dict[key]
    else:
        inner_start = n_outer * args.n_inner_loop
    
    # 获取当前archive中最大的fitness值
    max_fitness = 0
    for sol in archive:
        if 'fitness' in sol:
            try:
                fitness = extract_median(sol['fitness'])
                max_fitness = max(max_fitness, fitness)
            except:
                continue

    # for n_inner in range(inner_start, (n_outer + 1) * args.n_inner_loop):
    for n_inner in range(inner_start, inner_start + args.n_inner_loop):
        for solution in archive:
            if "generation" in solution and solution['generation'] == n_inner + 1 and solution['label'] == "inner_loop" and "fitness" in solution:
                continue
        print(f"============Inner Loop {n_inner + 1} for key {key}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.update_model)
        except Exception as e:
            print(f"During LLM generate new solution for key {key}:")
            print(e)
            n_inner -= 1
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list, _ = evaluate_forward_fn_key(args, next_solution["code"], key, args.train_dir_path)
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print(f"During evaluation for key {key}:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.update_model)
                except Exception as e:
                    print(f"During LLM generate new solution for key {key}:")
                    print(e)
                    continue
        if not acc_list:
            n_inner -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n_inner + 1
        next_solution['label'] = "inner_loop"

        if 'debug_thought' in next_solution: 
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        file_path = os.path.join(dir_path, f"{key}.json")
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
        # 如果当前solution的fitness大于历史最大值,则提前结束内循环
        if extract_median(fitness_str) > max_fitness:
            break
        
    return archive


def process_key_outer_evaluation(key, n_outer, solution, dir_path, archive, acc_list_dic, is_reflect):
    if key == "all":
        return None
    
    try:
        acc_list, fail_examples_list = evaluate_forward_fn_key(args, solution["code"], key, args.train_dir_path)
        acc_list_dic[key][solution['name']] = acc_list
    except Exception as e:
        print(f"During evaluating archive for key {key}:")
        print(e)
        return None

    fitness_str = bootstrap_confidence_interval(acc_list)
    
    solution_copy = copy.deepcopy(solution)
    solution_copy['fitness'] = fitness_str
    solution_copy['generation'] = n_outer + 1
    solution_copy['label'] = "outer_loop"
    if is_reflect:
        solution_copy['special_label'] = "fail_reflexion"
    
    archive[key].append(solution_copy)
    
    # save results
    file_path = os.path.join(dir_path, f"{key}.json")
    with open(file_path, 'w') as json_file:
        json.dump(archive[key], json_file, indent=4)
    
    return acc_list, fail_examples_list


def process_key_evaluation(args, key, sol, adaptation_archive, acc_list_dic, adaptation_file_path):
    print(f"Evaluating {key}...")
    try:
        if sol['generation'] == "initial":
            acc_list_dic[key], _ = evaluate_forward_fn_key(args, sol["code"], key, args.test_dir_path)
        else:
            # 随机采样六道题
            task_dsc = random.sample(load_all_examples(args.test_dir_path)[key], 6)
            task_dsc = "\n".join([
                f"##The task case {i+1} is following:\n{task['context']}"
                for i, task in enumerate(task_dsc)
            ])

            system_prompt, prompt = get_adaptation_prompt(sol, task_dsc)
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            next_solution = get_json_response_from_gpt_reflect(msg_list, args.update_model)
            acc_list_dic[key], _ = evaluate_forward_fn_key(args, next_solution["code"], key, args.test_dir_path)
            fitness_str = bootstrap_confidence_interval(acc_list_dic[key])

            next_solution['task'] = key
            next_solution['generation'] = sol['generation']
            next_solution['label'] = sol['label']
            if "special_label" in sol:
                next_solution['special_label'] = sol['special_label']
            next_solution['test_fitness'] = fitness_str

            adaptation_archive.append(next_solution)

            with open(adaptation_file_path, 'w') as json_file:
                json.dump(adaptation_archive, json_file, indent=4)
    except Exception as e:
        print(f"Error processing key {key}: {e}")
        acc_list_dic[key] = []
    return acc_list_dic[key]


def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive_{args.version}", "all_io.json")
    eval_file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive_{args.version}_evaluate_io.json")
    adaptation_file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive_{args.version}_evaluate_adaptation_io.json")
    
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    key_list = list(load_all_examples(args.train_dir_path).keys())

    adaptation_archive = []

    for sol in archive:
        if any(eval_sol['name'] == sol['name'] for eval_sol in eval_archive):
            continue
        print(f"current_gen: {sol['generation']}, current_name: {sol['name']}")
        
        acc_list_dic = {}
        with ThreadPoolExecutor(max_workers=len(key_list)) as executor:
            futures = {
                executor.submit(process_key_evaluation, args, key, sol, adaptation_archive, acc_list_dic, adaptation_file_path): key 
                for key in key_list
            }
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    acc_list_dic[key] = future.result()
                except Exception as e:
                    print(f"Error in future for key {key}: {e}")
                    acc_list_dic[key] = []

        acc_list = []
        for key in key_list:
            if key in acc_list_dic:
                acc_list.extend(acc_list_dic[key])
                
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)
    
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="drop_gpt4omini_results")
    parser.add_argument('--n_inner_loop', type=int, default=6)
    parser.add_argument('--n_outer_loop', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=2)
    parser.add_argument('--version', type=str, default="v1")
    parser.add_argument('--train_dir_path', type=str, default="dataset/drop/validate")
    parser.add_argument('--test_dir_path', type=str, default="dataset/drop/test")
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-mini-2024-07-18',
                        choices=['gpt-4o-mini-2024-07-18', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4o-mini-20240718', 'gpt-4o-mini'])
    parser.add_argument('--update_model',
                        type=str,
                        default='gpt-4.1',
                        choices=['gpt-4.1', 'gpt-4.1-20250414', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4o-mini-20240718', 'gpt-4o-mini'])

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
