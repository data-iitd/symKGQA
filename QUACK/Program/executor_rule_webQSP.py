from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
from utils.find_closest_webqsp import nearest_neigh, nearest_kv
import os
import sys
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm
import pandas as pd

"""
For convenience of implementation, in this rule-based execution engine,
all locating functions (including And Or) return (entity_ids, facts), 
even though some of them do not concern facts.
So that we can always use `entity_ids, _ = dependencies[0]` to store outputs.
"""
constrains = {                          # dependencies, inputs, returns, function
    # functions for locating entities
    'FindAll': [0, 0],                  # []; []; [(entity_ids, facts)]; get all ids of entities and concepts 
    'Find': [0, 1],                     # []; [entity_name]; [(entity_ids, facts)]; get ids for the given name
    'FilterConcept': [1, 1],            # [entity_ids]; [concept_name]; [(entity_ids, facts)]; filter entities by concept
    'FilterStr': [1, 2],                # [entity_ids]; [key, value]; [(entity_ids, facts)]
    'FilterNum': [1, 3],                # [entity_ids]; [key, value, op]; [(entity_ids, facts)]; op should be '=','>','<', or '!='
    'FilterYear': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'FilterDate': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'QFilterStr': [1, 2],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value]; [(entity_ids, facts)]; filter by facts
    'QFilterNum': [1, 3],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterYear': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterDate': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'Relate': [1, 2],                   # [entity_ids]; [predicate, direction]; [(entity_ids, facts)]; entity number should be 1
    
    # functions for logic
    'And': [2, 0],                      # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], intersection
    'Or': [2, 0],                       # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], union

    # functions for query
    'What': [1, 0],                     # [entity_ids]; []; [entity_name]; get its name, entity number should be 1
    'Count': [1, 0],                    # [entity_ids]; []; [count]
    'SelectBetween': [2, 2],            # [entity_ids_1, entity_ids_2]; [key, op]; [entity_name]; op is 'greater' or 'less', entity number should be 1
    'SelectAmong': [1, 2],              # [entity_ids]; [key, op]; [entity_name]; op is 'largest' or 'smallest'
    'QueryAttr': [1, 1],                # [entity_ids]; [key]; [value]; get the attribute value of given attribute key, entity number should be 1
    'QueryAttrUnderCondition': [1, 3],  # [entity_ids]; [key, qualifier_key, qualifier_value]; [value]; entity number should be 1
    'VerifyStr': [1, 1],                # [value]; [value]; [bool]; check whether the dependency equal to the input
    'VerifyNum': [1, 2],                # [value]; [value, op]; [bool];
    'VerifyYear': [1, 2],               # [value]; [value, op]; [bool];
    'VerifyDate': [1, 2],               # [value]; [value, op]; [bool];
    'QueryRelation': [2, 0],            # [entity_ids_1, entity_ids_2]; []; [predicate]; get the predicate between two entities, entity number should be 1
    'QueryAttrQualifier': [1, 3],       # [entity_ids]; [key, value, qualifier_key]; [qualifier_value]; get the qualifier value of the given attribute fact, entity number should be 1
    'QueryRelationQualifier': [2, 2],   # [entity_ids_1, entity_ids_2]; [predicate, qualifier_key]; [qualifier_value]; get the qualifier value of the given relation fact, entity number should be 1
}

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['word_idx_to_token'] = invert_dict(vocab['word_token_to_idx'])
    vocab['function_idx_to_token'] = invert_dict(vocab['function_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

class RuleExecutor(object):
    def __init__(self, entities, entity_name_to_ids, entity_name_to_ids2):
        # print('load kb')
        self.entities = entities
        self.idx = -1
        self.end = False
        self.lastEntitySet = []

        # replace adjacent space and tab in name
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = entity_name_to_ids
        self.entity_name_to_ids2 = entity_name_to_ids2

    def forward(self, program, inputs, idx, 
                ignore_error=False, show_details=False):
        memory = []
        self.idx = idx
        self.end = False
        self.lastEntitySet = []
        try:
            # infer the dependency based on the function definition
            dependency = []
            branch_stack = []
            for i, p in enumerate(program):
                if p in {'<START>', '<END>', '<PAD>'}:
                    dep = [0, 0]
                elif p in {'Find'}:
                    dep = [0, 0]
                    branch_stack.append(i - 1)
                else:
                    dep = [i-1, 0]
                dependency.append(dep)

            count_step = 0        
            for p, dep, inp in zip(program, dependency, inputs):
                if (count_step == len(program)-2):
                    self.end = True
                count_step += 1
                if p == '<START>':
                    res = None
                elif p == '<END>':
                    break
                else:
                    if(p == "QueryAttr"):
                        p = "Relate"
                    print(p)
                    func = getattr(self, p)
                    res = func([memory[d] for d in dep], inp)
                memory.append(res)
                if show_details:
#                     print(p, dep, inp)
                    print(res)
            
            return str(memory[-1])
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise
                
    def _parse_key_value(self, key, value, typ=None):
        if typ is None:
            if(key in self.key_type.keys()):
                typ = self.key_type[key]
            else:
                typ = 'string'
        if typ=='string':
            value = ValueClass('string', value)
        elif typ=='quantity':
            if ' ' in value:
                vs = value.split()
                v = vs[0]
                unit = ' '.join(vs[1:])
            else:
                v = value
                unit = '1'
            value = ValueClass('quantity', float(v), unit)
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                value = ValueClass('date', date(y, m, d))
            else:
                value = ValueClass('year', int(value))
        return value
    
    def isId(self, entities):
        try:
            for id_ in entities:
                if(id_ is None):
                    continue
                if(id_ not in self.entities.keys()):
                    return False
            return True
        except:
            return False

    def Find(self, dependencies, inputs):
        name = inputs[0]
        ## inputs[0] not in self.entity_name_to_ids then get_similar_entity_concept   
        if(name not in self.entity_name_to_ids):
            if(name not in self.entity_name_to_ids2):
                name = nearest_neigh(name, 'e', self.idx)
                entity_ids = [self.entity_name_to_ids[name]]
            else:
                entity_ids = [self.entity_name_to_ids2[name]]
        else:         
            entity_ids = [self.entity_name_to_ids[name]]
            
        if(len(entity_ids) != 0):
            self.lastEntitySet = entity_ids
        else:
            entity_ids = self.lastEntitySet    

        if(self.end == True):
            out_str = ""
            for i in range(len(entity_ids)):
                id_ = entity_ids[i]
                out_str += id_ + "|"
            return out_str
            
        return (entity_ids, None)
    
    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        res_ids = []
        res_facts = []
        list_attribute = []

        for i in entity_ids:
            for attr_key, attr_val in self.entities[i]['relations'].items():
                k = attr_key
                list_attribute.append(k)
                if k==tgt_key:
                    for val in attr_val:
                        v = val[0]
                        if v == tgt_value:
                            res_ids.append(i)
                            res_facts.append(attr_key)
                            break
                    
        if(res_ids == []):
            tgt_key = nearest_kv(tgt_key, 'attributes', list_attribute, self.idx)
            for i in entity_ids:
                for attr_key, attr_val in self.entities[i]['relations'].items():
                    k = attr_key
                    if k==tgt_key:
                        for val in attr_val:
                            v = val[0]
                            if v == tgt_value:
                                res_ids.append(i)
                                res_facts.append(attr_key)
                                break

        if(len(res_ids) != 0):
            self.lastEntitySet = res_ids
        else:
            res_ids = self.lastEntitySet

        if(self.end == True):
            out_str = ""
                
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                out_str += id_ + "|"
            return out_str

        return (res_ids, res_facts)
    
    def FilterStr(self, dependencies, inputs):
        try:
            entity_ids, _ = dependencies[0]
        except:
            entity_ids = self.lastEntitySet

        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        if(len(inputs) >= 3):
            key, value, op = inputs[0], inputs[1], inputs[2]
        else:
            key, value, op = inputs[0], inputs[1], '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')
    
    def Relate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        res_ids = []

        for id_ in entity_ids:
            if(id_ is None):
                continue
            else:
                out_ids = self.RelateHelper(id_, inputs)
                if(len(out_ids[0])):
                    res_ids += out_ids[0]
                    
        if(len(res_ids) != 0):
            self.lastEntitySet = res_ids
        else:
            res_ids = self.lastEntitySet

        if(self.end == True):
            out_str = ""
            
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                out_str += id_ + "|"
            return out_str
        
        
        return (res_ids, None)

    def RelateHelper(self, entity_id, inputs):
        predicate = inputs[0]
        res_ids = []
        list_relation = []
        if entity_id in self.entities:
            rel_infos = self.entities[entity_id]['relations']
        
        ## if predicate not in rel_info then take nearest using llm    
        for rel_info in rel_infos.keys():
            list_relation.append(rel_info)
            if rel_info == predicate:
                value = rel_infos[rel_info]
                for v in value:
                    res_ids.append(v[0])

        if(res_ids == []):
            predicate = nearest_kv(predicate, 'relation', list_relation, self.idx)
            res_ids = []

            if entity_id in self.entities:
                rel_infos = self.entities[entity_id]['relations']

            ## if predicate not in rel_info then take nearest using llm    
            for rel_info in rel_infos.keys():
                if rel_info == predicate:
                    value = rel_infos[rel_info]
                    for v in value:
                        res_ids.append(v[0])

        return (res_ids, None)


    def What(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids) or len(entity_ids) == 0):
            entity_ids = self.lastEntitySet

        name = ""
            
        for id_ in entity_ids:
            if(id_ is None):
                continue
            else:
                name += id_ + "|"
        if(len(name)):
            name = name[:-1]
        return name


def get_program(inp):
    program = inp['program']
    func = []
    inputs = []
    dep = []
    for i in range(len(program)):
        func.append(program[i]['function'])
        inputs.append(program[i]['inputs'])
        dep.append(program[i]['dependencies'])
    return func, inputs, dep

def get_pred_program(predicted):
    clean_prediction = []
    programs = []
    inputs = []
    for i in range(len(predicted)):
        steps = predicted[i].split('Step# ')[1:]
        result = []
        func = []
        inps = []
        for step in steps:
            try:
                result.append(": ".join(step.split(': ')[1:]).strip())
                temp = result[-1]
                flag = False
                start = 0
                out = []
                count = 0
                for j in range(len(temp)):
                    if(not flag and temp[j] == ','):
                        out.append(temp[start:j].strip())
                        start = j+1
                    elif(temp[j] == "("):
                        count += 1
                        if(count == 1):
                            out.append(temp[start:j].strip())
                            start = j+1
                    elif(temp[j] == ")"):
                        if(count == 1):
                            break
                        else:
                            count -= 1
                    elif(temp[j] == '"'):
                        flag = not flag
                out.append(temp[start:j].strip())
                func.append(out[0])
                if(out[0] in ["Find", "FilterConcept", "QueryAttr", "VerifyStr"]):
                    inps.append([", ".join(out[1:])])
                else:
                    inps.append(out[1:])
            except:
                print("error get_pred_program at idx", i)
#                 raise
                continue
        clean_prediction.append(result)
        programs.append(func)
        inputs.append(inps)

    return clean_prediction, programs, inputs


def getErrorIndex(log_file):
    f = open(log_file, 'r')
    logs = f.readlines()
    f.close()
    out = []
    idx = 0
    for i in range(len(logs)):
        line = logs[i]
        if(line == '-----------------------------------\n'):
            result = logs[i-1].split()
            if(result[0] != "Matched:"):
                out.append(idx)
            idx += 1
    return out

def getErrorIndexMeta(log_file):
    f = open(log_file, 'r')
    logs = f.readlines()
    f.close()
    out = []
    idx = 0
    for i in range(len(logs)):
        line = logs[i]
        if(line == '-----------------------------------\n'):
            result = logs[i-1].split()
            if(result[0] != "ground"):
                out.append(idx)
            idx += 1
    return out


def processKB(kb):
    entity_name_to_ids = {}
    entity_name_to_ids2 = {}
    entities = {}

    mapping = pd.read_csv('path of entity id to name mapping file1',sep='\t', header=None)
    mapping2 = pd.read_csv('path of entity id to name mapping file2')
    
    for i in range(len(mapping)):
        id_ = '.'.join(mapping[0][i].strip('/').split('/'))
        entity_name_to_ids[mapping[1][i]] = id_
        entities[id_] = {'name': mapping[1][i], 'relations': {}, 'raw_relations': {}}
        
    for i in range(len(mapping2)):
        entity_name_to_ids2[mapping2["name"][i]] = mapping2["id_"][i]
        
    id_ = 1
    error = 0
    for line in kb:
        line = line[:-1]
        kb_elements = line.strip().split('\t')
        try:
            if kb_elements[0] not in entities.keys():
                id_ = kb_elements[0]
                entity_name_to_ids[id_] = id_
                entities[id_] = {'name': id_, 'relations': {}, 'raw_relations': {}}
            
            if kb_elements[2] not in entities.keys():
                id_ = kb_elements[2]
                entity_name_to_ids[id_] = id_
                entities[id_] = {'name': id_, 'relations': {}, 'raw_relations': {}}
            
            enitiy_id = kb_elements[0]
            if(kb_elements[1].split('.')[-1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]].append([kb_elements[2], 'forward'])
                entities[enitiy_id]['raw_relations'][kb_elements[1]].append([kb_elements[2], 'forward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]] = [[kb_elements[2], 'forward']]
                entities[enitiy_id]['raw_relations'][kb_elements[1]] = [[kb_elements[2], 'forward']]
                    
            enitiy_id = entity_name_to_ids[kb_elements[2]]
            if(kb_elements[1].split('.')[-1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]].append([kb_elements[0], 'backward'])
                entities[enitiy_id]['raw_relations'][kb_elements[1]].append([kb_elements[0], 'backward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]] = [[kb_elements[0], 'backward']]
                entities[enitiy_id]['raw_relations'][kb_elements[1]] = [[kb_elements[0], 'backward']]
        except:
            error = 1
    
    return entity_name_to_ids, entities, entity_name_to_ids2

def extractFinalAnswer(filename):
    f = open(filename, "r")
    val = json.load(f)
    f.close()
    answers = []
    for i in range(len(val['Questions'])):
        answer = []
        for j in range(len(val['Questions'][i]['Parses'])):
            for k in range(len(val['Questions'][i]['Parses'][j]['Answers'])):
                answer.append(val['Questions'][i]['Parses'][j]['Answers'][k]['AnswerArgument'])
        answer = list(set(answer))
        answers.append(answer)
    return answers

def main():
    matched = 0
    inp_file = sys.argv[1]
    ground_file = sys.argv[2]
    kb_path = sys.argv[3]
    
    f = open(inp_file, 'r')
    gen = f.readlines()
    f.close()
    _, programs, inputs_all = get_pred_program(gen)

    f = open(kb_path, "r")
    kb = f.readlines()
    f.close()
    entity_name_to_ids, entities, entity_name_to_ids2 = processKB(kb)
    
    ground_answers = extractFinalAnswer(ground_file)

    for i in tqdm(range(len(programs))):
        print('idx:', i)
        try:
            idx = i
            rule_executor = RuleExecutor(entities, entity_name_to_ids, entity_name_to_ids2)
            if(len(programs[i]) > 4):
                program = programs[i][:3]
                inputs = inputs_all[i][:3]
            else:
                program = programs[i]
                inputs = inputs_all[i]
            program.insert(0, '<START>')
            program.append('<END>')
            inputs.insert(0, [])
            inputs.append([])
            pred = rule_executor.forward(program, inputs, idx, ignore_error=False, show_details=False)
            pred = list(set(pred.split('|')))
               
            ground_answer = ground_answers[i]
            flag = False
            
            intermediate_results = []
            for j in range(len(pred)):
                curr_pred = pred[j]
                if(curr_pred == ''):
                    continue

                if(entities[curr_pred]['name'] == curr_pred):
                    relations = entities[curr_pred]['relations']
                    for k, v in relations.items():
                        for z in range(len(v)):
                            intermediate_results.append(v[z][0])
            
            pred += intermediate_results
            pred = list(set(pred))
                    
            
            for j in range(len(pred)):
                if pred[j] in ground_answer:
                    print("Matched:", pred)
                    matched += 1
                    flag = True
                    break
            if(not flag):
                print("Not matched!", pred, ground_answer)

        except Exception as e:
            print('Error! ', e)
        print('-----------------------------------')
    return 0

if __name__ == '__main__':
    main()