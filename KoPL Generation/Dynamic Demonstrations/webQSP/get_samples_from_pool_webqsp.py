import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import json

top_100_samples = [(1894, 1205), (2447, 1204), (1287, 1148), (1776, 1137), (174, 1136), (974, 1133), (133, 1127), (347, 1105), (2227, 1098), (1296, 1095), (250, 1093), (2782, 1088), (68, 1085), (2663, 1082), (774, 1081), (1315, 1073), (2387, 1073), (1294, 1067), (2546, 1067), (2993, 1066), (1956, 1065), (2656, 1065), (2260, 1064), (388, 1064), (1607, 1060), (1047, 1059), (2774, 1059), (2646, 1057), (3039, 1056), (1107, 1053), (2431, 1052), (726, 1050), (1609, 1048), (828, 1047), (1622, 1046), (1679, 1046), (1748, 1042), (2284, 1040), (43, 1040), (1226, 1039), (606, 1037), (2094, 1036), (2080, 1034), (2767, 1031), (2505, 1028), (513, 1028), (1109, 1027), (676, 1027), (1387, 1026), (1716, 1024), (985, 1024), (1591, 1022), (1075, 1021), (1492, 1021), (1699, 1017), (1088, 1017), (2392, 1017), (921, 1016), (716, 1016), (1164, 1013), (1926, 1012), (2126, 1011), (2270, 1010), (890, 1009), (2811, 1007), (801, 1006), (162, 1004), (1142, 1001), (1721, 1000), (87, 999), (2620, 999), (2879, 998), (1909, 997), (1563, 997), (1360, 997), (933, 997), (2149, 997), (2438, 992), (1743, 991), (2402, 990), (2148, 988), (932, 988), (2012, 988), (2594, 987), (49, 987), (2308, 987), (2955, 986), (124, 986), (805, 986), (1050, 985), (2140, 984), (818, 984), (2286, 983), (1640, 982), (2007, 981), (1528, 979), (2334, 977), (2709, 977), (983, 976), (1407, 975)]
model = SentenceTransformer('all-distilroberta-v1')
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./")

f = open("../data/webqsp/WebQSP.train.json")
train_data = json.load(f)['Questions']
f.close()

train_samples = [train_data[i]['RawQuestion'] for (i, j) in top_100_samples]
sentence_embeddings = model.encode(train_samples)

collection = client.get_or_create_collection("top-100-documents-webqsp_new", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i,_ in top_100_samples], # unique for each doc
    )


def create_icl(ques_ids, train_data):
    ques_prompt = ""
    i = 1
    for ques_id in ques_ids:
        ques = train_data[int(ques_id)]['RawQuestion']
        entity = train_data[int(ques_id)]['Parses'][0]['TopicEntityName']
        chain = train_data[int(ques_id)]['Parses'][0]['InferentialChain']
        constraints = train_data[int(ques_id)]['Parses'][0]['Constraints']
        if chain:
            ques_prompt += f"Training Example {i}:\nQuestion: {ques} Entities: ['{entity}']. The steps to solve this question are:\nOutput:\nStep# 1: Find({entity})\n"
            for j in range(len(chain)):
                relation = chain[j].split('.')[-1]
                if relation.endswith('_s'):
                    ques_prompt += f"Step# {j+2}: Relate({''.join(relation.split('_'))}, forward)\n"
                else:
                    ques_prompt += f"Step# {j+2}: Relate({' '.join(relation.split('_'))}, forward)\n"

            if constraints:
                for k in range(len(constraints)):
                    relation = constraints[k]['NodePredicate'].split('.')[-1]
                    entityname = constraints[k]['EntityName']
                    if relation != "from" and relation != "to":
                        ques_prompt += f"Step# {j+k+3}: FilterStr({' '.join(relation.split('_'))}, {entityname})\n"
            ques_prompt += "Done\n\n"
            i += 1
    ques_prompt += "Test Example:\nQuestion: "
    return ques_prompt
