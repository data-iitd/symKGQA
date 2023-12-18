import json
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings("ignore")

api_key='' #add api key here

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
- Or function is always come by at least after two Find or FindAll functions.
- And function is always come by at least after two Find or FindAll functions.
- If Concept is None in question, then you is not allowed to generate FilterConcept function.

Function format:
- FunctionName:
Description - <Description of function>, 
Functional Inputs - <Inputs to the current step function from previous step function output >, 
Textual Inputs - <Textual inputs to the current step function>, 
Outputs - <Outputs of the current step function>

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
Outputs - Entities

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

Training Example 1:
Question: What is the connection between A Serious Man to Ireland (the one whose nominal GDP is 239389340720.488 United States dollar)? Entities: ['A Serious Man', 'Ireland'], Concepts: None. The steps to solve this question are:
Output:
Step 1: Find(A Serious Man)
Step 2: Find(Ireland)
Step 3: FilterNum(nominal GDP, 239389340720.488 United States dollar, =)
Step 4: QueryRelation()
Done

Training Example 2:
Question: Which first-level administrative country subdivision established post-1829 covers the biggest area?  Entities: None, Concepts: ['first-level administrative country subdivision']. The steps to solve this question are:
Output:
Step 1: FindAll()
Step 2: FilterYear(inception, 1829, >)
Step 3: FilterConcept(first-level administrative country subdivision)
Step 4: SelectAmong(area, largest)
Done

Training Example 3:
Question: What is the ISNI of John Broome (the one born in 1738-01-01)?  Entities: ['John Broome'], Concepts: None. The steps to solve this question are:
Output:
Step 1: Find(John Broome)
Step 2: FilterDate(date of birth, 1738-01-01, =)
Step 3: QueryAttr(ISNI)
Done

Training Example 4:
Question: Does the sovereign state that has a diplomatic relation with Malaysia (the subject of this statement is East Timor–Malaysia relations), have the CIVICUS Monitor country entry of saint-lucia?  Entities: ['Malaysia'], Concepts: ['sovereign state']. The steps to solve this question are:
Output:
Step 1: Find(Malaysia)
Step 2: Relate(diplomatic relation, forward)
Step 3: QFilterStr(statement is subject of, East Timor–Malaysia relations)
Step 4: FilterConcept(sovereign state)
Step 5: QueryAttr(CIVICUS Monitor country entry)
Step 6: VerifyStr(saint-lucia)
Done

Training Example 5:
Question: What is the umber of episodes in TV series with Twitter username ThomasFriends (the subscription number of this statement is 15947)?  Entities: None, Concepts: ['television series']. The steps to solve this question are:
Output:
Step 1: FindAll()
Step 2: FilterStr(Twitter username, ThomasFriends)
Step 3: QFilterNum(number of subscribers, 15947, =)
Step 4: FilterConcept(television series)
Step 5: QueryAttr(number of episodes)
Done

Training Example 6:
Question: When was born the person that was nominated for Tony Award for Best Actor in a Musical in 1967?  Entities: ['Tony Award for Best Actor in a Musical'], Concepts: ['human']. The steps to solve this question are:
Output:
Step 1: Find(Tony Award for Best Actor in a Musical)
Step 2: Relate(nominated for, backward)
Step 3: QFilterYear(point in time, 1967, =)
Step 4: FilterConcept(human)
Step 5: QueryAttr(date of birth)
Done

Training Example 7:
Question: Does Pierce County that is located in Washington or Grays Harbor County have less area?  Entities: ['Washington', 'Pierce County', 'Grays Harbor County'], Concepts: None. The steps to solve this question are:
Output:
Step 1: Find(Washington)
Step 2: Relate(located in the administrative territorial entity, backward)
Step 3: Find(Pierce County)
Step 4: And()
Step 5: Find(Grays Harbor County)
Step 6: SelectBetween(area, less)
Done

Training Example 8:
Question: How many researchers are the occupation of Aristotle or practice motivational speaking?  Entities: ['Aristotle', 'motivational speaking'], Concepts: ['researcher']. The steps to solve this question are:
Output:
Step 1: Find(Aristotle)
Step 2: Relate(occupation, forward)
Step 3: FilterConcept(researcher)
Step 4: Find(motivational speaking)
Step 5: Relate(practiced by, forward)
Step 6: FilterConcept(researcher)
Step 7: Or()
Step 8: Count()
Done

Training Example 9:
Question: Is the nominal GDP of Guinea-Bissau over 69000000 United States dollars on the date 1996-01-01?  Entities: ['Guinea-Bissau'], Concepts: None. The steps to solve this question are:
Output:
Step 1: Find(Guinea-Bissau)
Step 2: QueryAttrUnderCondition(nominal GDP, point in time, 1996-01-01)
Step 3: VerifyNum(69000000 United States dollar, >)
Done

Training Example 10:
Question: Which university has fewer students, George Washington University or University of Hamburg?  Entities: ['George Washington University', 'University of Hamburg'], Concepts: None. The steps to solve this question are:
Output:
Step 1: Find(George Washington University)
Step 2: Find(University of Hamburg)
Step 3: SelectBetween(students count, less)
Done

Test Example:
Question: 
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


f = open("data/KQAPro.IID/val.json")
data = json.load(f)
f.close()


def run(dataset, init_prompt):
    model_name = "models/text-bison-001"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(len(dataset))):
        if (i+1)%100 == 0:
            time.sleep(60)
        entities, concepts = get_entities_concepts(dataset[i]['program'])
        input_text = init_prompt + dataset[i]['question'] + f' Entities: {entities}.' + f' Concepts: {concepts}.' + ' The steps to solve this question are:\nOutput:'
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
                # output = result["candidates"][2]['output'].strip()
                # gen_steps_3.append(output)
            else:
                gen_steps_2.append("NA")
                gen_steps_3.append("NA")
            # print((gen_steps_1, gen_steps_2, gen_steps_3))
        else:
            gen_steps_1.append("NA")
            gen_steps_2.append("NA")
            gen_steps_3.append("NA")
        f_out = open("palm_output/palm_kqapro_sampling_1.json", 'w')
        json.dump(gen_steps_1, f_out, indent=4)
        f_out.close()
        f_out = open("palm_output/palm_kqapro_sampling_2.json", 'w')
        json.dump(gen_steps_2, f_out, indent=4)
        f_out.close()
        # f_out = open("palm_output/out_val_run_3_new_palm.json", 'w')
        # json.dump(gen_steps_3, f_out, indent=4)
        # f_out.close()
    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)