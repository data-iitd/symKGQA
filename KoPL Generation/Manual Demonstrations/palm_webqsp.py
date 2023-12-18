import json
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings("ignore")


api_key='' #add API key here

def generate_output(inp_text, model_name):
    json_payload={
                "model": model_name,  # this is where you can change the model used
                "prompt":{"text":inp_text},
                "candidate_count":2,    #change it to 1 for greedy decoding
                "temperature":0.3,      #change it to 0 for greedy decoding
                "top_k":30,             #comment for greedy decoding
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
- FilterStr
Description - Filter entities with an attribute condition of string type in the knowledge graph. Return entities and corresponding facts in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Key, Value), 
Outputs - (Entities, Facts)

Training Examples:

Training Example 1:
Question: what form of government does afghanistan have? Entities: ['Afghanistan']. The steps to solve this question are:
Output:
Step# 1: Find(Afghanistan)
Step# 2: Relate(form of government, forward)
Done

Training Example 2:
Question: what book did charles darwin write on evolution? Entities: ['Charles Darwin']. The steps to solve this question are:
Output:
Step# 1: Find(Charles Darwin)
Step# 2: Relate(works written, forward)
Step# 3: FilterStr(notable types, Book)
Step# 4: FilterStr(subjects, Evolution)
Done

Training Example 3:
Question: what kind of money does the uk use? Entities: ['United Kingdom']. The steps to solve this question are:
Output:
Step# 1: Find(United Kingdom)
Step# 2: Relate(currency used, forward)
Done

Training Example 4:
Question: who is facebook's founder? Entities: ['Facebook, Inc.']. The steps to solve this question are:
Output:
Step# 1: Find(Facebook, Inc.)
Step# 2: Relate(founders, forward)
Done

Training Example 5:
Question: what is mexico city time zone? Entities: ['Mexico City']. The steps to solve this question are:
Output:
Step# 1: Find(Mexico City)
Step# 2: Relate(time zones, forward)
Done

Training Example 6:
Question: where does liz mcclarnon live? Entities: ['Liz McClarnon']. The steps to solve this question are:
Output:
Step# 1: Find(Liz McClarnon)
Step# 2: Relate(places lived, forward)
Step# 3: Relate(location, forward)
Done

Training Example 7:
Question: who is ruling tunisia now? Entities: ['Tunisia']. The steps to solve this question are:
Output:
Step# 1: Find(Tunisia)
Step# 2: Relate(governing officials, forward)
Step# 3: Relate(office holder, forward)
Step# 4: FilterStr(basic title, Acting President)
Done

Training Example 8:
Question: who plays jacob black in the twilight movies? Entities: ['Twilight']. The steps to solve this question are:
Output:
Step# 1: Find(Twilight)
Step# 2: Relate(starring, forward)
Step# 3: Relate(actor, forward)
Step# 4: FilterStr(character, Jacob Black)
Done

Training Example 9:
Question: what countries are on the mediterranean sea? Entities: ['Mediterranean Sea']. The steps to solve this question are:
Output:
Step# 1: Find(Mediterranean Sea)
Step# 2: Relate(adjoins, forward)
Step# 3: Relate(adjoins, forward)
Step# 4: FilterStr(notable types, Country)
Done

Training Example 10:
Question: what year did tut became king? Entities: ['Tutankhamun']. The steps to solve this question are:
Output:
Step# 1: Find(Tutankhamun)
Step# 2: Relate(reign, forward)
Step# 3: Relate(start, forward)
Done


Test Example:
Question: '''


f = open("data/WebQSP/data/WebQSP.test.json")
data = json.load(f)['Questions']
f.close()

def run(dataset, init_prompt):
    model_name = "models/text-bison-001"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(len(dataset))):
        if (i+1)%100 == 0:
            time.sleep(60)
        ques = dataset[i]['RawQuestion']
        entity = dataset[i]['Parses'][0]['TopicEntityName']
        input_text = init_prompt + ques + f". Entities: ['{entity}']." + ' The steps to solve this question are:\nOutput:'
        result = generate_output(input_text, model_name)
        if "candidates" in result.keys() and len(result["candidates"]) > 0:
            output = result["candidates"][0]['output'].strip()
            gen_steps_1.append(output)
            # if len(result["candidates"]) == 3:
            #     output = result["candidates"][1]['output'].strip()
            #     gen_steps_2.append(output)
            #     output = result["candidates"][2]['output'].strip()
            #     gen_steps_3.append(output)
            if len(result["candidates"]) == 2:
                output = result["candidates"][1]['output'].strip()
                gen_steps_2.append(output)
            else:
                gen_steps_2.append("NA")
            # print((gen_steps_1, gen_steps_2, gen_steps_3))
        else:
            gen_steps_1.append("NA")
            gen_steps_2.append("NA")
            gen_steps_3.append("NA")
        f_out = open("palm_output/palm_webqsp_sampling1.json", 'w')
        json.dump(gen_steps_1, f_out, indent=4)
        f_out.close()
        f_out = open("palm_output/palm_webqsp_sampling2.json", 'w')
        json.dump(gen_steps_2, f_out, indent=4)
        f_out.close()
        # f_out = open("palm_output/out_val_run_3_new_palm.json", 'w')
        # json.dump(gen_steps_3, f_out, indent=4)
        # f_out.close()
        # print(input_text)
    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)