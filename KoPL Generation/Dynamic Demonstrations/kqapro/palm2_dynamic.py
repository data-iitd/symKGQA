import json
from tqdm import tqdm
import time
import requests
from get_samples_from_pool import *
import warnings
warnings.filterwarnings("ignore")


api_key='ADD API KEY'
def generate_output(inp_text, model_name):
    json_payload={
                "model": model_name,  # this is where you can change the model used
                "prompt":{"text":inp_text},
                "candidate_count":2,
                "temperature":0.3,
                "top_k":30,
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
- FindAll
Description - Return all entities in the knowledge graph, 
Functional Inputs - None, 
Textual Inputs - None, 
Outputs - Entities

- Find
Description - Return all entities with the given name in the knowledge graph, 
Functional Inputs - None, 
Textual Inputs - name, 
Outputs - Entities

- FilterConcept
Description - Find those belonging to the given concept in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - concept name, 
Outputs - Entities

- FilterStr
Description - Filter entities with an attribute condition of string type in the knowledge graph. Return entities and corresponding facts in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Key, Value), 
Outputs - (Entities, Facts)

- FilterNum
Description - Similar to FilterStr, except that the attribute type is number in the knowledge graph, Functional Inputs - Entities, 
Textual Inputs - (Key, Value, Operation), 
Outputs - (Entities, Facts)

- FilterYear
Description - Similar to FilterStr, except that the attribute type is year in the knowledge graph, Functional Inputs - Entities, 
Textual Inputs - (Key, Value, Operation), 
Outputs - (Entities, Facts)

- FilterDate
Description - Similar to FilterStr, except that the attribute type is date in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Key, Value, Operation), 
Outputs - (Entities, Facts)

- QFilterStr
Description - Filter entities and corresponding facts with a qualifier condition of string type in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Qualifier Key, Qualifier Value), 
Outputs - (Entities, Facts)

- QFilterNum
Description - Similar to QFilterStr, except that the qualifier type is number in the knowledge graph,
Functional Inputs - Entities, 
Textual Inputs - (Qualifier Key, Qualifier Value, Operation), 
Outputs - (Entities, Facts)

- QFilterYear
Description - Similar to QFilterStr, except that the qualifier type is year in the knowledge graph,
 Functional Inputs - Entities, 
Textual Inputs - (Qualifier Key, Qualifier Value, Operation), 
Outputs - (Entities, Facts)

- QFilterDate
Description - Similar to QFilterStr, except that the qualifier type is date in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Qualifier Key, Qualifier Value, Operation), 
Outputs - (Entities, Facts)

- Relate
Description - Find entities that have a specific relation with the given entity in the knowledge graph, Functional Inputs - Entity, 
Textual Inputs - (Predicate, Direction), 
Outputs - (Entities, Facts)

- And
Description - Return the intersection of two entity sets in the knowledge graph, 
Functional Inputs - (Entities, Entities), 
Textual Inputs - None, 
Outputs - Entities

- Or
Description - Return the union of two entity sets in the knowledge graph, 
Functional Inputs - (Entities, Entities), 
Textual Inputs - None, 
Outputs - Entities, 

- QueryName
Description - Return the entity name in the knowledge graph, 
Functional Inputs - Entity, 
Textual Inputs - None, 
Outputs - string

- Count
Description - Return the number of entities in the knowledge graph, 
Functional Inputs - Entity, 
Textual Inputs - None, 
Outputs - number

- QueryAttr
Description - Return the attribute value of the entity in the knowledge graph, 
Functional Inputs - Entity, 
Textual Inputs - Key, 
Outputs - Value

- QueryAttrUnderCondition
Description - Return the attribute value whose corresponding fact should satisfy the qualifier condition in the knowledge graph, 
Functional Inputs - Entity, 
Textual Inputs - (Key, Qualifier Key, Qualifier Value), 
Outputs - Value

- QueryRelation 
Description - Return the relation between two entities in the knowledge graph, 
Functional Inputs - (Entity, Entity), 
Textual Inputs - None, 
Outputs - Predicate

- SelectBetween
Description - From the two entities, find the one whose attribute value is greater or less and return its name in the knowledge graph, 
Functional Inputs - (Entity, Entity), 
Textual Inputs - (Key, Operation), 
Outputs - string

- SelectAmong
Description - From the entity set, find the one whose attribute value is the largest or smallest in the knowledge graph, 
Functional Inputs - Entities, 
Textual Inputs - (Key, Operation), 
Outputs - string

- VerifyStr
Description - Return whether the output of QueryAttr or QueryAttrUnderCondition and the given value are equal as string in the knowledge graph, 
Functional Inputs - Value, 
Textual Inputs - Value, 
Outputs - boolean

- VerifyNum
Description - Return whether the two numbers satisfy the condition in the knowledge graph, 
Functional Inputs - Value, 
Textual Inputs - (Value, Operation), 
Outputs - boolean

- VerifyYear
Description - Return whether the two years satisfy the condition in the knowledge graph, 
Functional Inputs - Value, 
Textual Inputs - (Value, Operation), 
Outputs - boolean

- VerifyDate
Description - Return whether the two dates satisfy the condition in the knowledge graph, 
Functional Inputs - Value, 
Textual Inputs - (Value, Operation), 
Outputs - boolean

- QueryAttrQualifier
Description - Return the qualifier value of the fact (Entity, Key, Value) in the knowledge graph, 
Functional Inputs - Entity, 
Textual Inputs - (Key, Value, Qualifier Key), 
Outputs - Qualifier Value

- QueryRelationQualifier
Description - Return the qualifier value of the fact (Entity, Pred, Entity) in the knowledge graph, 
Functional Inputs - (Entity, Entity), 
Textual Inputs - (Predicate, Qualifier Key),
Outputs - Qualifier Value

Training Examples:

'''

def get_entities_concepts(program):
    entities = []
    concepts = []
    for prog in program:
        if prog['function'] == "Find":
            entities.extend(prog['inputs'])
        elif prog['function'] == "FilterConcept":
            concepts.extend(prog['inputs'])
    if len(entities) > 0 and len(concepts) > 0:
        return entities, concepts
    elif len(entities) > 0 and len(concepts) == 0:
        return entities, None
    elif len(entities) == 0 and len(concepts) > 0:
        return None, concepts
    elif len(entities) == 0 and len(concepts) == 0:
        return None, None


f = open("../data/KQAPro.IID/val.json")
data = json.load(f)
f.close()

f = open("../data/KQAPro.IID/val.json")
data = json.load(f)
f.close()


f = open("../data/KQAPro.IID/train.json")
train_data = json.load(f)
f.close()

def run(dataset, init_prompt):
    model_name = "models/text-bison-001"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(len(dataset))):
        if (i+1)%100 == 0:
            time.sleep(60)
        target_question = ' | '.join(data[i]['program'][j]['function'] for j in range(len(data[i]['program'])))
        target_embeddings = model.encode(target_question)
        results = collection.query(
            query_embeddings=target_embeddings.tolist(),
            n_results=15
        )
        ids = results.get('ids')[0]
        prompt = create_icl(ids, train_data)
        entities, concepts = get_entities_concepts(dataset[i]['program'])
        input_text = init_prompt + prompt + dataset[i]['question'] + f' Entities: {entities}.' + f' Concepts: {concepts}.' + ' The steps to solve this question are:\nOutput:'
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
        f_out = open("Add path of output file 1.json", 'w')
        json.dump(gen_steps_1, f_out, indent=4)
        f_out.close()
        f_out = open("Add path of output file 2.json", 'w')
        json.dump(gen_steps_2, f_out, indent=4)
        f_out.close()

    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)