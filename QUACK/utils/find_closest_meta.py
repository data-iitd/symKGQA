import torch
import json
import os
from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertModel, LlamaTokenizer, LlamaForCausalLM, StoppingCriteriaList , AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from transformers import LlamaTokenizer, LlamaForCausalLM, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
import time
import guidance

class PriorityQueue(object):
	def __init__(self):
		self.queue = []
		self.key = []

	def get_queue(self):
		return self.queue
    
	def get_key(self):
		return self.key

	def __str__(self):
		return ' '.join([str(i) for i in self.queue])

	# for checking if the queue is empty
	def isEmpty(self):
		return len(self.queue) == 0

	# for inserting an element in the queue
	def insert(self, data):
		self.queue.append(data)
		self.key.append(data[1])

	# for popping an element based on Priority
	def delete(self):
		try:
			min_val = 0
			for i in range(len(self.queue)):
				if self.queue[i][0] < self.queue[min_val][0]:
					min_val = i
			item = self.queue[min_val]
			del self.queue[min_val]
			return item
		except IndexError:
			print()
			exit()


            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BERT model and tokenizer
model_bert = SentenceTransformer('all-distilroberta-v1').to(device)

# model_path = 'openlm-research/open_llama_3b_v2'
model_path = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

print('Model loading done!')


def processKB(kb):
    entity_name_to_ids = {}
    entities = {}
    id_ = 1
    for line in kb:
        line = line[:-1]
        kb_elements = line.split('|')
        if kb_elements[0] not in entity_name_to_ids.keys():
            entity_name_to_ids[kb_elements[0]] = id_
            entities[id_] = {'name': kb_elements[0], 'relations': {kb_elements[1]: [[kb_elements[2], 'forward']]}}
            id_ += 1
        else:
            enitiy_id = entity_name_to_ids[kb_elements[0]]
            if(kb_elements[1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1]].append([kb_elements[2], 'forward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1]] = [[kb_elements[2], 'forward']]
            
        if kb_elements[2] not in entity_name_to_ids.keys():
            entity_name_to_ids[kb_elements[2]] = id_
            entities[id_] = {'name': kb_elements[2], 'relations': {kb_elements[1]: [[kb_elements[0], 'backward']]}}
            id_ += 1
        else:
            enitiy_id = entity_name_to_ids[kb_elements[2]]
            if(kb_elements[1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1]].append([kb_elements[0], 'backward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1]] = [[kb_elements[0], 'backward']]
    
    return entity_name_to_ids, entities

f = open("kb file path", "r")
kb_file = f.readlines()
f.close()
entity_name_to_ids, kb = processKB(kb_file)

f = open("val/test set file path")
data = json.load(f)
f.close()

class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False


def generate_output(inp_text):
    input_ids = tokenizer(inp_text, return_tensors="pt").input_ids.to(device)
    stopping_criteria_list = StoppingCriteriaList([
        _SentinelTokenStoppingCriteria(
            sentinel_token_ids=tokenizer(
                "\n",
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to("cuda"),
            starting_idx=input_ids.shape[-1])
    ])
    output = model.generate(
        input_ids=input_ids, max_new_tokens=2048, stopping_criteria=stopping_criteria_list
    )
    return tokenizer.decode(output[0])


def calculate_cosine_similarity_opt_new(currentQuery, queryListFinal):
    embeddings = model_bert.encode(queryListFinal)
    currentQueryEmbeedings = model_bert.encode(currentQuery)
    similarities = cosine_similarity(torch.from_numpy(currentQueryEmbeedings), torch.from_numpy(embeddings))
    closest = similarities.argsort(descending=True)
    sorted_list = []
    for ind in closest[:10]:
        sorted_list.append((similarities[ind].item(), queryListFinal[ind]))
    return sorted_list

def get_closest_entity_opt(word1, nbas):
    cos_sim = []
    part = len(nbas)//8
    if(part == 0):
        cos_sim.append(calculate_cosine_similarity_opt_new(word1, nbas))
    else:     
        for i in range(8):
            start = i*part
            end = (i+1)*part
            if(i == 7):
                end = len(nbas)
            cos_sim.append(calculate_cosine_similarity_opt_new(word1, nbas[start:end]))
    myQueue  = PriorityQueue()
    for i in range(len(cos_sim)):
        for ele in cos_sim[i]:
            if(ele[1] not in myQueue.get_key()):
                myQueue.insert(ele)
            if(len(myQueue.get_queue()) > 10):
                myQueue.delete()
    top_k_entities = []
    while not myQueue.isEmpty():
        top = myQueue.delete()
        top_k_entities.append(top[1])
    top_k_entities.reverse()
    return top_k_entities
      
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
                result.append(step.split(': ')[1].strip())
                temp = result[-1]
                flag = False
                start = 0
                out = []
                for i in range(len(temp)):
                    if(not flag and temp[i] == ','):
                        out.append(temp[start:i].strip())
                        start = i+1
                    elif(temp[i] == "("):
                        out.append(temp[start:i].strip())
                        start = i+1
                    elif(temp[i] == ")"):
                        break
                    elif(temp[i] == '"'):
                        flag = not flag
                out.append(temp[start:i].strip())
                func.append(out[0])
                inps.append(out[1:])
            except:
#                 print(i)
                continue
        clean_prediction.append(result)
        programs.append(func)
        inputs.append(inps)

    return clean_prediction, programs, inputs

def select_entitiy(top_k, question, word):
    if(len(top_k) == 0):
        print('top_k is empty!')
        return None
    print("topK: ", top_k)

    guidance.llm = guidance.llms.Transformers(model = model, tokenizer = tokenizer)

    prompt = "From below entity list, select only top 3 entities which are most similar and belong to similar entity extracted from question. Entity list = [" + ', '.join(top_k)  + "]"
    sentence = question + " Entity extracted from Question - "  + word
    options = top_k
    program = guidance('''
            {{prompt}}
            Input : {{que}}
            Output : {{select "answer" options=valid_options}}
            ''')

    executed_program = program(prompt = prompt,  
                                    que = sentence,
                                valid_options = options)
    
    ans = executed_program['answer']
    print('ans ', ans)
    return ans.strip()


def nearest_kv(predicate, type_, list_relation, idx):
    print('nearest_kv', predicate)
    list_relation = list(set(list_relation))
    print('input_list: ', list_relation)
    if(predicate in list_relation):
        print('ans ', predicate)
        return predicate
    top_k = get_closest_entity_opt(predicate, list_relation)
    print("topK: ", top_k)
    return select_entitiy(top_k, data[idx]['nlq'], predicate)

def nearest_neigh(word, type_, idx):
    print('nearest_neigh', word)
    nbas = []
    if(type_ == 'e'):
        for key, value in kb.items():
            nbas.append(value['name'])
    else:
        for key, value in kb.items():
            nbas.append(value['name'])
    nbas = list(set(nbas))
    if(word in nbas):
        print('ans ', word)
        return word
    top_k = get_closest_entity_opt(word, nbas)
    print("topK: ", top_k)
    return select_entitiy(top_k, data[idx]['nlq'], word)

def nearest_neigh_by_id(word, ids, type_, idx):
    print('nearest_neigh_by_id', word)
    top_k = []
    if(type_ == 'e'):
        for key, value in kb.items():
            if(key in ids):
                top_k.append(value['name'])
    else:
        for key, value in kb.items():
            if(key in ids):
                top_k.append(value['name'])
    top_k = list(set(top_k))
    if(word in top_k):
        print('ans ', word)
        return word
    top_k = get_closest_entity_opt(word, top_k)
    print("topK: ", top_k)
    return select_entitiy(top_k, data[idx]['nlq'], word)
