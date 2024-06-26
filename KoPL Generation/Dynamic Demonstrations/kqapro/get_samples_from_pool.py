import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
from kopl_to_program import *


top_100_samples = [81630, 94327, 81590, 94333, 65668, 93375, 79656, 84736, 94368, 93943, 94338, 93313, 93535, 48630, 94350, 93306, 58436, 56100, 50369, 94295, 21852, 82135, 94361, 94038, 60062, 85177, 94334, 94180, 85788, 32265, 91441, 81068, 85659, 24909, 94119, 94311, 57826, 80895, 89199, 92070, 94136, 923, 85156, 87634, 94070, 67512, 87493, 94372, 16051, 56537, 84781, 93932, 12499, 27057, 93951, 94216, 93813, 50814, 93709, 93001, 93524, 68280, 94344, 94364, 67809, 75408, 92080, 94065, 92785, 93223, 94370, 94328, 68625, 38134, 94336, 93096, 90111, 57064, 94351, 94363, 89359, 49481, 94164, 94200, 89699, 79454, 94209, 93080, 26967, 84985, 93279, 93767, 56402, 93047, 82843, 93823, 86763, 76476, 93830, 90014, 88840, 61805, 59195, 39091, 54225, 14956, 12826, 42761, 53110, 35049]
model = SentenceTransformer('all-distilroberta-v1')
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./")

f = open("../data/train.json")
train_data = json.load(f)
f.close()

def get_entities_concepts(program):
    entities = []
    concepts = []
    for prog in program:
        if prog['function'] == "Find":
            entities.extend(prog['inputs'])
        elif prog['function'] == "FilterConcept":
            concepts.extend(prog['inputs'])
    if len(entities) > 0 and len(concepts) > 0:
        return list(set(entities)), list(set(concepts))
    elif len(entities) > 0 and len(concepts) == 0:
        return list(set(entities)), None
    elif len(entities) == 0 and len(concepts) > 0:
        return None, list(set(concepts))
    elif len(entities) == 0 and len(concepts) == 0:
        return None, None

train_samples = []
for i in top_100_samples:
    program = train_data[i]['program']
    program_steps = ""
    for j in range(len(program)):
            program_steps += f"Step {j+1}: {program[j]['function']}({', '.join(program[j]['inputs'])}) | "
    train_samples.append(program_steps)

sentence_embeddings = model.encode(train_samples)
collection = client.get_or_create_collection("top-110-documents-program-steps", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i in top_100_samples], # unique for each doc
    )

def create_icl(ques_ids, train_data):
    ques_prompt = ""
    i = 1
    for ques_id in ques_ids:
        ques = train_data[int(ques_id)]['question']
        program = train_data[int(ques_id)]['program']
        entities, concepts = get_entities_concepts(program)
        ques_prompt += f"Training Example {i}:\nQuestion: {ques} Entities: {entities}. Concepts: {concepts}. The steps to solve this question are:\nOutput:\n"
        for j in range(len(program)):
            ques_prompt += f"Step {j+1}: {program[j]['function']}({', '.join(program[j]['inputs'])})\n"
        ques_prompt += "Done\n\n"
        i += 1
    ques_prompt += "Test Example:\nQuestion: "
    return ques_prompt
