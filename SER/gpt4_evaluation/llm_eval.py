import os
from tqdm import tqdm
import json
import re
#from llm_components import get_oai_completion_gpt_unified
import argparse
import pandas as pd
import json
from statistics import mean
from cloudgpt_aoai import get_openai_client

def llm_eval_prompt(question, answer, gptversion=4):
    system_prompt='''
    Review the user's question and the corresponding response using the additive 5-pointscoring system described below. Points are accumulated based on the satisfaction of each criterion:
    - Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
    - Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
    - Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an Al Assistant or if it has elements typically found in blogs or search results.
    - Grant a fourth point if the response is clearly written from an Al Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
    - Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, refecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.
    
    After examining the user's instruction and the response, please first give the total score. Then provide a brief explanation of your total score, up to 100 words. Output with the following format:
    Score: <total score>
    Evaluation evidence: <your brief explanation here>
    
    Remember to assess from the Al Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we'llsystematically attribute points based on the outlined criteria.
    '''

    answers_prompt='''
    <Question>: {question}
    <response>: {answer}
    '''

    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": answers_prompt.format(question=question, answer=answer)
        }
    ]

    #output=get_oai_completion_gpt_unified(message_list, gptversion)
    client = get_openai_client()
    output = client.chat.completions.create(model="gpt-4o-20240806",
                               messages=message_list,
                               temperature=0.7,
                               max_tokens=1024,
                               top_p=0.95,
                               frequency_penalty=0,
                               presence_penalty=0,)
    print(output.choices[0].message)

    return output.choices[0].message
    

if __name__ == '__main__':
    paths = [
        #'llama3_8b_full_dataset_ppo_test.json',
        #'llama3_8b_self_label_ppo_test.json',
        #'llama3_8b_sft_ppo_test.json',
        #'llama3_3b_policy_SFT_inference.json',
        #'llama32_3b_SFT_inference.json'
    ]
    for path in paths:
        output_list = []
        with open('C:/workspace/data/ultrafeedback/Inference_RM_test_winrate/' + path, 'r') as file:
            res = file.read()
    
        res = json.loads(res)

        for i in tqdm(range(len(res))):
            review=llm_eval_prompt(res[i]['question'], res[i]['response'])
            #print(review.content)
            
            output_list.append(dict(
                question=res[i]['question'],
                answer=res[i]['response'],
                review=review.content,
            ))

        with open('C:/workspace/data/ultrafeedback/Inference_RM_test_winrate/gpt_evaluation_' + path, 'w') as f:
            json.dump(output_list, f, indent=2)
