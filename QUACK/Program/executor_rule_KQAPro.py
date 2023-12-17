from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
from QUACK.utils.find_closest_kqapro import nearest_neigh, nearest_kv, nearest_neigh_by_id
import os
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm
import sys
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
    def __init__(self, vocab, kb_json):
        self.vocab = vocab
        # print('load kb')
        kb = json.load(open(kb_json))
        self.concepts = kb['concepts']
        self.entities = kb['entities']
        self.idx = -1
        self.end = False
        self.lastEntitySet = []

        # replace adjacent space and tab in name
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = defaultdict(list)
        for ent_id, ent_info in self.entities.items():
            self.entity_name_to_ids[ent_info['name']].append(ent_id)
        self.concept_name_to_ids = defaultdict(list)
        for con_id, con_info in self.concepts.items():
            self.concept_name_to_ids[con_info['name']].append(con_id)

        self.concept_to_entity = defaultdict(set)
        self.entity_to_concept = defaultdict(set)
        for ent_id in self.entities:
            for c in self._get_all_concepts(ent_id): # merge entity into ancestor concepts
                self.concept_to_entity[c].add(ent_id)
                self.entity_to_concept[ent_id].add(c)
        self.concept_to_entity = { k:list(v) for k,v in self.concept_to_entity.items() }
        self.entity_to_concept = { k:list(v) for k,v in self.entity_to_concept.items() }

        self.key_type = {}
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
            
       
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk in rel_info['qualifiers']:
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    rel_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]

        # some entities may have relations with concepts, we add them into self.concepts for visiting convenience
        for ent_id in self.entities:
            for rel_info in self.entities[ent_id]['relations']:
                obj_id = rel_info['object']
                if obj_id in self.concepts:
                    if 'relations' not in self.concepts[obj_id]:
                        self.concepts[obj_id]['relations'] = []
                    self.concepts[obj_id]['relations'].append({
                        'relation': rel_info['relation'],
                        'predicate': rel_info['relation'], # predicate
                        'direction': 'forward' if rel_info['direction']=='backward' else 'backward',
                        'object': ent_id,
                        'qualifiers': rel_info['qualifiers'],
                        })

    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupport value type')
        return result

    def _get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['subclassOf'] # instanceOf

    def _get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self._get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for c in self.concepts[con_id]['subclassOf']:  # instaceOf
                q.put(c)
        return ancestors

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
                elif p in {'FindAll', 'Find'}:
                    dep = [0, 0]
                    branch_stack.append(i - 1)
                elif p in {'And', 'Or', 'SelectBetween', 'QueryRelation', 'QueryRelationQualifier'}:
                    if(p == "QueryRelation" and len(inputs) >= 1 and inputs[i][0] != ''):
                        dep = [i-1, 0]
                    elif(p == "SelectBetween" and branch_stack[-1] == 0):
                        program[i] = "SelectAmong"
                        dep = [i-1, 0]
                    else:
                        dep = [branch_stack[-1], i-1]
                        branch_stack = branch_stack[:-1]
                        
                else:
                    if(p == "Relate" and len(inputs[i]) == 1 and inputs[i][0] == ''):
                        dep = [branch_stack[-1], i-1]
                        branch_stack = branch_stack[:-1]
                    else:
                        dep = [i-1, 0]
                dependency.append(dep)

            count_step = 0        
            for p, dep, inp in zip(program, dependency, inputs):
                if (count_step == len(program)-2):
                    # print('i ', i)
                    self.end = True
                count_step += 1
                if p == '<START>':
                    res = None
                elif p == '<END>':
                    break
                else:
                    print(p)
                    functional_inp = [memory[d] for d in dep]
                    if(p == "QFilterStr" and functional_inp[0][1] == None):
                        p = "FilterStr"
                        print("Executing: ", p)
                    elif(p == "QFilterNum" and functional_inp[0][1] == None):
                        p = "FilterNum"
                        print("Executing: ", p)
                    elif(p == "QFilterYear" and functional_inp[0][1] == None):
                        p = "FilterYear"
                        print("Executing: ", p)
                    elif(p == "QFilterDate" and functional_inp[0][1] == None):
                        p = "FilterDate"
                        print("Executing: ", p)
                    elif(p == "QueryRelation" and len(inp) >= 1 and inp[0] != ''):
                        p = "Relate"
                        print("Executing: ", p)
                    elif(p == "Relate" and len(inp) == 1 and inp[0] == ''):
                        p = "QueryRelation"
                        print("Executing: ", p)
                    elif(p in ["QueryAttrQualifier", "QueryAttrUnderCondition"] and len(inp) >= 3 and isOp(inp[2])):
                        p = "FilterStr"
                        print("Executing: ", p)
                    
                    if(p == "FilterStr" and functional_inp[0][1] != None):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QFilterStr"
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                    elif(p == "FilterNum" and functional_inp[0][1] != None):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QFilterNum"
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                    elif(p == "FilterYear" and functional_inp[0][1] != None):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QFilterYear"
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                    elif(p == "FilterDate" and functional_inp[0][1] != None):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QFilterDate"
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                    elif(p == "QueryAttrUnderCondition"):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QueryAttrQualifier"
                            inp[1], inp[2] = inp[2], inp[1]
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                    elif(p == "QueryAttrQualifier"):
                        try:
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)
                        except:
                            p = "QueryAttrUnderCondition"
                            inp[1], inp[2] = inp[2], inp[1]
                            print("Executing: ", p)
                            func = getattr(self, p)
                            res = func([memory[d] for d in dep], inp)        
                    else:
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
                if(id_ not in self.entities.keys() and id_ not in self.concepts.keys()):
                    return False
            return True
        except:
            return False

    def FindAll(self, dependencies, inputs):
        entity_ids = list(self.entities.keys())
        self.lastEntitySet = entity_ids
        return (entity_ids, None)

    def Find(self, dependencies, inputs):
        name = inputs[0]
        ## inputs[0] not in self.entity_name_to_ids then get_similar_entity_concept
        if(name not in self.entity_name_to_ids and name not in self.concept_name_to_ids):
            name = nearest_neigh(name, 'e', self.idx)
        entity_ids = self.entity_name_to_ids[name]
        
        # concept may appear in some relations
        if name in self.concept_name_to_ids: 
            entity_ids += self.concept_name_to_ids[name]

        if(self.end == True):
            out_str = ""
            for i in range(len(entity_ids)):
                id_ = entity_ids[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        self.lastEntitySet = entity_ids    
        return (entity_ids, None)

    def FilterConcept(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet
    
        concept_name = inputs[0]
         ## inputs[0] not in self.concept_name_to_ids then get_similar_entity_concept\
        flag = False
        if(concept_name not in self.concept_name_to_ids):
            flag = True
            concept_name = nearest_neigh(concept_name, 'c', self.idx)

        concept_ids = self.concept_name_to_ids[concept_name]
        entity_ids_2 = []
        for i in concept_ids:
            entity_ids_2 += self.concept_to_entity[i]
        entity_ids = list(set(entity_ids) & set(entity_ids_2))
        if(entity_ids == [] and flag):
            ids = []
            entity_ids, _ = dependencies[0]
            if(not self.isId(entity_ids)):
                entity_ids = self.lastEntitySet
            concept_name = inputs[0]
            for id_ in entity_ids:
                ids += self.entity_to_concept[id_]
            ids = list(set(ids))
            concept_name = nearest_neigh_by_id(concept_name, ids, 'c', self.idx)
            concept_ids = self.concept_name_to_ids[concept_name]
            entity_ids_2 = []
            for i in concept_ids:
                entity_ids_2 += self.concept_to_entity[i]
            entity_ids = list(set(entity_ids) & set(entity_ids_2))

        if(self.end == True):
            out_str = ""
            for i in range(len(entity_ids)):
                id_ = entity_ids[i]
                name = ""
                if(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                elif(id_ in self.entities.keys()):
                        name = self.entities[id_]['name']
                out_str += name + " "
            return out_str.strip()

        self.lastEntitySet = entity_ids      
        return (entity_ids, None)

    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        raw_tag_value = tgt_value
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        list_attribute = []

        for i in entity_ids:
            for attr_info in self.entities[i]['attributes']:
                k, v = attr_info['key'], attr_info['value']
                list_attribute.append(k)
                if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                    res_ids.append(i)
                    res_facts.append(attr_info)
                    
        if(res_ids == []):
            tgt_key = nearest_kv(tgt_key, 'attributes', list_attribute, self.idx)
            tgt_value = self._parse_key_value(tgt_key, raw_tag_value, typ)
            for i in entity_ids:
                for attr_info in self.entities[i]['attributes']:
                    k, v = attr_info['key'], attr_info['value']
                    if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                        res_ids.append(i)
                        res_facts.append(attr_info)

        self.lastEntitySet = res_ids

        if(self.end == True):
            out_str = ""
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        return (res_ids, res_facts)

    def FilterStr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        if(len(inputs) >= 3):
            key, value, op = inputs[0], inputs[1], inputs[2]
        else:
            key, value, op = inputs[0], inputs[1], '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')

    def FilterNum(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'quantity')

    def FilterYear(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'year')

    def FilterDate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'date')

    def _filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, op, typ):
        raw_tgt_value = tgt_value
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        list_qualifier = []
        # if tgt_key not in qualifiers, take nearest using llm
        for i, f in zip(entity_ids, facts):
            for qk, qvs in f['qualifiers'].items():
                list_qualifier.append(qk)
                if qk == tgt_key:
                    for qv in qvs:
                        if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                            res_ids.append(i)
                            res_facts.append(f)
        if(res_ids == []):
            tgt_key = nearest_kv(tgt_key, 'qualifiers', list_qualifier, self.idx)
            tgt_value = self._parse_key_value(tgt_key, raw_tgt_value, typ)
            for i, f in zip(entity_ids, facts):
                for qk, qvs in f['qualifiers'].items():
                    if qk == tgt_key:
                        for qv in qvs:
                            if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                                res_ids.append(i)
                                res_facts.append(f)

        self.lastEntitySet = res_ids

        if(self.end == True):
            out_str = ""
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        return (res_ids, res_facts)

    def QFilterStr(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], '='
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'string')

    def QFilterNum(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'quantity')

    def QFilterYear(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'year')

    def QFilterDate(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'date')

    def Relate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        entity_id = entity_ids[0]
        predicate, direction = inputs[0], inputs[1]
        res_ids = []
        res_facts = []
        list_relation = []
        if entity_id in self.entities:
            rel_infos = self.entities[entity_id]['relations']
        else:
            rel_infos = self.concepts[entity_id]['relations']
        ## if predicate not in rel_info then take nearest using llm    
        for rel_info in rel_infos:
            list_relation.append(rel_info['relation'])
            if rel_info['relation']==predicate:
                res_ids.append(rel_info['object'])
                res_facts.append(rel_info)

        
        if(res_ids == []):
            inputs[0] = nearest_kv(predicate, 'relation', list_relation, self.idx)
            # print('line360', inputs[0])
            entity_ids, _ = dependencies[0]
            if(not self.isId(entity_ids)):
                entity_ids = self.lastEntitySet

            entity_id = entity_ids[0]
            predicate, direction = inputs[0], inputs[1]
            res_ids = []
            res_facts = []
            if entity_id in self.entities:
                rel_infos = self.entities[entity_id]['relations']
            else:
                rel_infos = self.concepts[entity_id]['relations']
            ## if predicate not in rel_info then take nearest using llm    

            for rel_info in rel_infos:
                if rel_info['relation']==predicate:
                    res_ids.append(rel_info['object'])
                    res_facts.append(rel_info)

        self.lastEntitySet = res_ids

        if(self.end == True):
            out_str = ""
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        return (res_ids, res_facts)

    def And(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        try:
            entity_ids_2, _ = dependencies[1]
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1)):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2)):
            entity_ids_2 = self.lastEntitySet

        self.lastEntitySet = list(set(entity_ids_1) & set(entity_ids_2)) 

        if(self.end == True):
            out_str = ""
            for i in range(len(self.lastEntitySet)):
                id_ = self.lastEntitySet[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        return (list(set(entity_ids_1) & set(entity_ids_2)), None)

    def Or(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        try:
            entity_ids_2, _ = dependencies[1]
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1)):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2)):
            entity_ids_2 = self.lastEntitySet

        self.lastEntitySet = list(set(entity_ids_1) | set(entity_ids_2)) 

        if(self.end == True):
            out_str = ""
            for i in range(len(self.lastEntitySet)):
                id_ = self.lastEntitySet[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                elif(id_ in self.concepts.keys()):
                    name = self.concepts[id_]['name']
                out_str += name + " "
            return out_str.strip()

        return (list(set(entity_ids_1) | set(entity_ids_2)), None)

    def What(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        entity_id = entity_ids[0]
        name = self.entities[entity_id]['name']
        return name

    def QueryName(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        entity_id = entity_ids[0]
        name = self.entities[entity_id]['name']
        return name

    def Count(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        return len(entity_ids)

    def SelectBetween(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        try:
            entity_ids_2, _ = dependencies[1]
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1)):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2)):
            entity_ids_2 = self.lastEntitySet
        dependencies[0] = (entity_ids_1 + entity_ids_2, [])
        return self.SelectAmong(dependencies, inputs)

    def SelectAmong(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet
            
        key, op = inputs[0], inputs[1]
        candidates = []
        list_attribute = []
        comparators = [['greater', 'larger', 'longer', 'greatest', 'largest', 'longest', 'biggest'], ['less', 'smaller', 'shorter', 'smallest', 'shortest', 'least']]
        for i in entity_ids:
            if(i not in self.entities.keys()):
                continue
            flag = False
            for attr_info in self.entities[i]['attributes']:
                list_attribute.append(attr_info['key'])
                if key == attr_info['key']:
                    flag = True
                    v = attr_info['value']
            if(flag):  
                candidates.append((i, v))
        if(len(candidates) == 0):
            key = nearest_kv(key, 'attributes', list_attribute, self.idx)
            for i in entity_ids:
                if(i not in self.entities.keys()):
                    continue
                flag = False
                for attr_info in self.entities[i]['attributes']:
                    if key == attr_info['key']:
                        flag = True
                        v = attr_info['value']
                if(flag):  
                    candidates.append((i, v))

        sort = sorted(candidates, key=lambda x: x[1])
        i = sort[0][0] if (op in comparators[1]) else sort[-1][0]
        name = self.entities[i]['name']
        return name

    def QueryAttr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        entity_id = entity_ids[0]
        key = inputs[0]
        list_attribute = []
        flag = False
        
        for attr_info in self.entities[entity_id]['attributes']:
            list_attribute.append(attr_info['key'])
            if key == attr_info['key']:
                v = attr_info['value']
                flag = True
        if(not flag):
            key = nearest_kv(key, 'attributes', list_attribute, self.idx)
            for attr_info in self.entities[entity_id]['attributes']:
                if key == attr_info['key']:
                    v = attr_info['value']
        return v
    
    def QueryAttrUnderCondition(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet

        entity_id = entity_ids[0]
        key, qual_key, qual_value = inputs[0], inputs[1], inputs[2]
        qual_value = self._parse_key_value(qual_key, qual_value)
        list_attribute = []
        list_qualifier = []

        flag = False            
        for attr_info in self.entities[entity_id]['attributes']:
            list_attribute.append(attr_info['key'])
            if key == attr_info['key']:
                flag = False
                for qk, qvs in attr_info['qualifiers'].items():
                    if qk == qual_key:
                        for qv in qvs:
                            if qv.can_compare(qual_value) and comp(qv, qual_value, "="):
                                flag = True
                                break
                    if flag:
                        break
                if flag:
                    v = attr_info['value']
                    break

        if(not flag):
            key = nearest_kv(key, 'attributes', list_attribute, self.idx)
    
            for attr_info in self.entities[entity_id]['attributes']:
                if key == attr_info['key']:
                    for qk, qvs in attr_info['qualifiers'].items():
                        list_qualifier.append(qk)
            
            qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)
            qual_value = self._parse_key_value(qual_key, inputs[2])

            for attr_info in self.entities[entity_id]['attributes']:
                if key == attr_info['key']:
                    flag = False
                    for qk, qvs in attr_info['qualifiers'].items():
                        if qk == qual_key:
                            for qv in qvs:
                                if qv.can_compare(qual_value) and comp(qv, qual_value, "="):
                                    flag = True
                                    break
                        if flag:
                            break
                    if flag:
                        v = attr_info['value']
                        break
                    
        return v

    def _verify(self, dependencies, value, op, typ):
        attr_value = dependencies[0]
        value = self._parse_key_value(None, value, typ)
        if attr_value.can_compare(value) and comp(attr_value, value, op):
            if(not self.end):
                return (self.lastEntitySet, None)
            answer = 'yes'
        else:
            answer = 'no'
        return answer

    def VerifyStr(self, dependencies, inputs):
        value, op = inputs[0], '='
        return self._verify(dependencies, value, op, 'string')
        

    def VerifyNum(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'quantity')

    def VerifyYear(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'year')

    def VerifyDate(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'date')

    def QueryRelation(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        try:
            entity_ids_2, _ = dependencies[1]
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1)):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2)):
            entity_ids_2 = self.lastEntitySet

        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']
        p = None
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                p = rel_info['relation']
        return p

    def QueryAttrQualifier(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        if(not self.isId(entity_ids)):
            entity_ids = self.lastEntitySet
        
        entity_id = entity_ids[0]
        key, value, qual_key = inputs[0], inputs[1], inputs[2]
        value = self._parse_key_value(key, value)
        list_attribute = []
        list_qualifier = []
       
        for attr_info in self.entities[entity_id]['attributes']:
            list_attribute.append(attr_info['key'])
            if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                comp(attr_info['value'], value, '='):
                for qk, qvs in attr_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]
        
         ## if key not in attributes then take nearest using llm  
        
        key = nearest_kv(key, 'attributes', list_attribute, self.idx)
        value = self._parse_key_value(key, inputs[1])
        
        for attr_info in self.entities[entity_id]['attributes']:
            if key == attr_info['key']:
                for qk, qvs in attr_info['qualifiers'].items():
                    list_qualifier.append(qk)
        
        qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)
                
        for attr_info in self.entities[entity_id]['attributes']:
            list_attribute.append(attr_info['key'])
            if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                comp(attr_info['value'], value, '='):
                for qk, qvs in attr_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]

        return None

    def QueryRelationQualifier(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        try:
            entity_ids_2, _ = dependencies[1]
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1)):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2)):
            entity_ids_2 = self.lastEntitySet

        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        predicate, qual_key = inputs[0], inputs[1]

        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']

        list_relation = []
        list_qualifier = []

        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                list_relation.append(rel_info['relation'])
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]

        ## Nearest
        predicate = nearest_kv(predicate, 'relation', list_relation, self.idx)
        
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    list_qualifier.append(qk)
        
        qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)

        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                list_relation.append(rel_info['relation'])
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]

        return None


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
        steps = predicted[i].split('Step ')[1:]
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


def main():
    vocab = load_vocab('./preprocessed_kb/vocab.json')
    input_file_path = sys.argv[1]
    ground_file_path = sys.argv[2]


    val = json.load(open(ground_file_path))
    input_dir = './preprocessed_kb/'
    matched = 0

    f = open(input_file_path, 'r')
    gen = f.readlines()
    f.close()

    _, programs, inputs_all = get_pred_program(gen)

    for i in tqdm(range(len(val))):
        print('idx:', i)
        try:
            idx = i
            rule_executor = RuleExecutor(vocab, os.path.join(input_dir, 'kb.json'))
            program = programs[i]
            inputs = inputs_all[i]
            program.insert(0, '<START>')
            program.append('<END>')
            inputs.insert(0, [])
            inputs.append([])
            pred = rule_executor.forward(program, inputs, idx, ignore_error=False, show_details=False)
            if(pred != val[idx]['answer']):
                print("Not matched!", pred, val[idx]['answer'])
            else:
                print("Matched:", pred)
                matched += 1
        except Exception as e:
            print('Error! ', e)
        print('-----------------------------------')

    print('Accuracy: ', matched/len(val))
    return 0

if __name__ == '__main__':
    main()
