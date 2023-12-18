import json
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings("ignore")


api_key= "" #add api key

def generate_output(inp_text, model_name):
    json_payload={
                "model": model_name,  # this is where you can change the model used
                "prompt":{"text":inp_text},
                "candidate_count":2,
                "temperature":0.3,  #0 for greedy
                "top_k":30,         #comment for greedy
                "stop_sequences":['Done'],
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

Training Example 1:
Question: the actor Michael Sarrazin co-starred with who. Entities: ['Michael Sarrazin']. The steps to solve this question are:
Output:
Step# 1: Find(Michael Sarrazin)
Step# 2: Relate(starred actors,backward)
Step# 3: Relate(starred actors,forward)
Step# 4: What()
Done

Training Example 2:
Question: who directed the films written by Walter Wager. Entities: ['Walter Wager']. The steps to solve this question are:
Output:
Step# 1: Find(Walter Wager)
Step# 2: Relate(written by,backward)
Step# 3: Relate(directed by,forward)
Step# 4: What()
Done

Training Example 3:
Question: what genres do the movies written by Christine Bell fall under. Entities: ['Christine Bell']. The steps to solve this question are:
Output:
Step# 1: Find(Christine Bell)
Step# 2: Relate(written by,backward)
Step# 3: Relate(has genre,forward)
Step# 4: What()
Done

Training Example 4:
Question: who are the directors of the films written by Yasmina Reza. Entities: ['Yasmina Reza']. The steps to solve this question are:
Output:
Step# 1: Find(Yasmina Reza)
Step# 2: Relate(written by,backward)
Step# 3: Relate(directed by,forward)
Step# 4: What()
Done

Training Example 5:
Question: who are the actors in the movies written by Don Tait. Entities: ['Don Tait']. The steps to solve this question are:
Output:
Step# 1: Find(Don Tait)
Step# 2: Relate(written by,backward)
Step# 3: Relate(starred actors,forward)
Step# 4: What()
Done

Training Example 6:
Question: which directors co-directed films with David Lister. Entities: ['David Lister']. The steps to solve this question are:
Output:
Step# 1: Find(David Lister)
Step# 2: Relate(directed by,backward)
Step# 3: Relate(directed by,forward)
Step# 4: What()
Done

Training Example 7:
Question: the director of Captains of the Clouds also directed which movies. Entities: ['Captains of the Clouds']. The steps to solve this question are:
Output:
Step# 1: Find(Captains of the Clouds)
Step# 2: Relate(directed by,forward)
Step# 3: Relate(directed by,backward)
Step# 4: What()
Done

Training Example 8:
Question: who directed the movies written by Josephine Lawrence. Entities: ['Josephine Lawrence']. The steps to solve this question are:
Output:
Step# 1: Find(Josephine Lawrence)
Step# 2: Relate(written by,backward)
Step# 3: Relate(directed by,forward)
Step# 4: What()
Done

Training Example 9:
Question: what are the genres of the films starred by James Balog. Entities: ['James Balog']. The steps to solve this question are:
Output:
Step# 1: Find(James Balog)
Step# 2: Relate(starred actors,backward)
Step# 3: Relate(has genre,forward)
Step# 4: What()
Done

Training Example 10:
Question: who starred in the movies written by Matt Reeves. Entities: ['Matt Reeves']. The steps to solve this question are:
Output:
Step# 1: Find(Matt Reeves)
Step# 2: Relate(written by,backward)
Step# 3: Relate(starred actors,forward)
Step# 4: What()
Done


Test Example:
Question: '''

org_file_1 = open('data/MetaQA/2-hop/vanilla/qa_test.txt')
data = org_file_1.readlines()
org_file_1.close()


def run(dataset, init_prompt):

    model_name = "models/text-bison-001"
    gen_steps_1 = []
    gen_steps_2 = []

    for i in tqdm(range(len(dataset))):
        if (i+1)%100 == 0:
            time.sleep(60)
        ques = dataset[i].split('\t')[0]
        entity = ques[ques.find("[")+1:ques.find("]")]
        ques = ques.replace('[', '').replace(']', '')
        input_text = init_prompt + ques + f". Entities: ['{entity}']." + ' The steps to solve this question are:\nOutput:'
        result = generate_output(input_text, model_name)
        if "candidates" in result.keys() and len(result["candidates"]) > 0:
            output = result["candidates"][0]['output'].strip()
            gen_steps_1.append(output)
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
        f_out = open("palm_output/palm_metaqa_2hop_sampling1.json", 'w')
        json.dump(gen_steps_1, f_out, indent=4)
        f_out.close()
        f_out = open("palm_output/palm_metaqa_2hop_sampling2.json", 'w')
        json.dump(gen_steps_2, f_out, indent=4)
        f_out.close()
    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)