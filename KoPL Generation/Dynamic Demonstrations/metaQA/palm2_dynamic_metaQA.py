import json
from tqdm import tqdm

import time
import requests
from get_samples_from_pool_metaqa import *
import warnings
warnings.filterwarnings("ignore")

api_key='ADD API KEY'

def generate_output(inp_text, model_name):
    json_payload={
                "model": model_name,  # this is where you can change the model used
                "prompt":{"text":inp_text},
                "candidate_count":1,
                "temperature":0,
                # "top_k":30,
                "stop_sequences":['Step 17:', 'Done'],
                "max_output_tokens":800,
                "safetySettings": [ 
                { 
                    "category": "HARM_CATEGORY_DEROGATORY", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_VIOLENCE", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_SEXUAL", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_MEDICAL", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_DANGEROUS", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_TOXICITY", 
                    "threshold": "BLOCK_NONE" 
                },
            ]
                }
    url = f'https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key={api_key}'
    headers = {"Content-Type": "application/json"
                   }
    response = requests.post(url, headers=headers, json=json_payload, verify=False)
    res = response.json()
    return res

init_prompt = '''
Instruction -
Your have to follow below instruction to achieve an end task. Always validate the output using below instructions, and don't try to generate anything which you think is wrong.
- The task is to come up with the steps of functions for test example from the list of functions given in "List of Functions" section.
- Each function can take "functional inputs" which is the "output" from the previous step, and "textual inputs" which you have to generate from a given question. If textual input is None for any function then you don't have to generate any textual input for that function.
- To generate these steps you should match "output" of a current step function with "functional inputs" of the next step function.
- Use the training examples to understand the step generation process and stick only to the output format provided in the training examples. Do not generate any explanation text.
- Do not use entities and concepts outside of the list provided in each test question. If None is mentioned in concept in question then it means that their is no concept present in the test question and you can't generate any concept related function.
 - "Functional input" of current step can be subset of previous step "output", but can't be superset, for example if function input of current step is (entity, entity) then previous step output can't be entity only, it should be at least (entity, entity).
- Or function is always come by at least after two Find or FindAll functions.
- And function is always come by at least after two Find or FindAll functions.
- If Concept is None in question, then you is not allowed to generate FilterConcept function.
Function format:
- FunctionName:
Description - <Description of function>, Functional Inputs - <Inputs to the current step function from previous step function output >, Textual Inputs - <Textual inputs to the current step function>, Outputs - <Outputs of the current step function>
List of Functions:
- Find
Description - Return all entities with the given name in the knowledge graph,
Functional Inputs - None,
Textual Inputs - name,
Outputs - Entities
- Relate
Description - Find entities that have a specific relation with the given entity in the knowledge graph, Functional Inputs - Entity,
Textual Inputs - (Predicate, Direction),
Outputs - (Entities, Facts)

Training Examples:

'''


f = open('../data/metaQA/1-hop/vanilla/qa_test.txt')
test_data = f.readlines()
f.close()

f = open('../data/metaQA/1-hop/vanilla/qa_train.txt')
train_data = f.readlines()
f.close()

file = open('../data/metaQA/1-hop/vanilla/metaqa_train.json')
annotated_data = json.load(file)
file.close()

def run(dataset, init_prompt, annotated_data_dict):
    model_name = "models/text-bison-001"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(len(dataset))):
        target_question = test_data[i].split('\t')[0].replace('[', '').replace(']', '')
        target_embeddings = model.encode(target_question)
        results = collection.query(
            # query_texts=[target],
            query_embeddings=target_embeddings.tolist(),
            n_results=15
        )
        ids = results.get('ids')[0]
        prompt = create_icl(ids, train_data, annotated_data_dict)
        entity = target_question[target_question.find("[")+1:target_question.find("]")]
        target_question = target_question.replace('[', '').replace(']', '')
        input_text = init_prompt + prompt + target_question + f". Entities: ['{entity}']." + "The steps to solve this question are:\nOutput:"
        # print(input_text)
        result = generate_output(input_text, model_name)
        if "candidates" in result.keys() and len(result["candidates"]) > 0:
            output = result["candidates"][0]['output'].strip()
            gen_steps_1.append(output)
            if len(result["candidates"]) == 3:
                output = result["candidates"][1]['output'].strip()
                gen_steps_2.append(output)
                output = result["candidates"][2]['output'].strip()
                gen_steps_3.append(output)
            if len(result["candidates"]) == 2:
                output = result["candidates"][1]['output'].strip()
                gen_steps_2.append(output)
            else:
                gen_steps_2.append("NA")
                gen_steps_3.append("NA")
        else:
            gen_steps_1.append("NA")
            gen_steps_2.append("NA")
            gen_steps_3.append("NA")
        f_out = open("Add output file path", 'w')
        json.dump(gen_steps_1, f_out, indent=4)
        f_out.close()
    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(test_data, init_prompt, annotated_data_dict)\
