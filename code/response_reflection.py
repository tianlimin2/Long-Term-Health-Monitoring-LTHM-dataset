
"""# Test the response effect of memory reflection

1. Load data, load memory nodesnodes.json and embedding.json, load question-answering dataset QA.json
2. Recall retrieval mechanism that adjusts weight values ​​based on question categories
3. Respond based on retrieved related nodes
"""



import json
from datetime import datetime, timedelta
import openai
from openai import AzureOpenAI
import numpy as np

# OpenAI API Configuration - Chat Response
openai.api_type = "azure"
openai.api_version = '2023-05-15'
deployment_name = 'gpt-4'
openai.api_key = "ed4bec37b4ad4c55936bee64a302350e"
openai.api_base = "https://healingangels.openai.azure.com/"

client_chat = AzureOpenAI(
    api_key=openai.api_key,
    api_version=openai.api_version,
    azure_endpoint=openai.api_base,
)

# OpenAI API Configuration  - embedding
embedding_model = 'text-embedding-ada-002'
embedding_deployment = 'text-embedding-ada-002'
embedding_api_base = 'https://webgpt.openai.azure.com'
embedding_api_key = '0853dbeb2f0f46089839b1fde7d3d86e'

client_embedding = AzureOpenAI(
  api_key=embedding_api_key,
  api_version="2024-02-01",
  azure_endpoint=embedding_api_base
)

# Make a chat API request and return the content of the chat response
def get_chat_response(deployment_name, messages):
    response = client_chat.chat.completions.create(
        model=deployment_name,
        messages=messages
    )
    
    return response.choices[0].message.content

# Make an Embedding Vector API request and return the embedding vector of the text
def get_embeddings(texts, model="text-embedding-ada-002"):
    '''Encapsulate OpenAI's Embedding model interface'''
    data = client_embedding.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

def load_memory(file_path):
    try:
        with open(file_path, 'r') as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = {"nodes": []}
    return memory

def load_embeddings(file_path):
    try:
        with open(file_path, 'r') as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        embeddings = {}
    return embeddings



# Comprehensive search auxiliary function

def cos_sim(a, b):
    """Compute the cosine similarity between two vectors"""
    a = np.squeeze(a) 
    b = np.squeeze(b)  
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize_dict_floats(d):
    """Normalize the floating point values ​​in the dictionary to the range 0 to 1"""
    min_val = min(d.values())
    max_val = max(d.values())
    range_val = max_val - min_val

    if range_val == 0:
        for key in d:
            d[key] = 0.5
    else:
        for key in d:
            d[key] = (d[key] - min_val) / range_val
    return d

def top_highest_x_values(d, x):
    """Extract the first x key-value pairs with the highest values ​​from a dictionary"""
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:x])
# Recency function
def extract_recency(nodes, recency_decay=0.9):
    
    recency_vals = [(recency_decay ** (len(nodes) - i)) for i in range(len(nodes))]
    recency_out = {}
    for count, node in enumerate(nodes):
        recency_out[node['node_id']] = recency_vals[count]
    return recency_out

def extract_importance(nodes):
    
    importance_out = {}
    for node in nodes:
        importance_out[node['node_id']] = node['poignancy']
    return importance_out


# Calculate the similarity with the question embedding vector question embedding
def extract_relevance(nodes, question_embedding, embeddings):
    
    relevance_out = {}
    for node in nodes:
        try:
            node_embedding = embeddings[node['embedding_key']]
            relevance_out[node['node_id']] = cos_sim(node_embedding, question_embedding)
        except KeyError:
            
            print(f"Embedding not found for key: {node['embedding_key']}")
            relevance_out[node['node_id']] = 0 
    return relevance_out


def retrieve(question, embeddings, memory, n_count=3):
    retrieved = {}
    question_embedding = get_embeddings([question])[0]
    nodes = sorted(memory['nodes'], key=lambda x: x['last_accessed'])

    recency_out = normalize_dict_floats(extract_recency(nodes))
    importance_out = normalize_dict_floats(extract_importance(nodes))
    relevance_out = normalize_dict_floats(extract_relevance(nodes, question_embedding, embeddings))

    gw = [0.33, 0.33, 0.33]
    master_out = {}
    detailed_scores = {}
    for key in recency_out.keys():
        total_score = (gw[0] * recency_out[key] + gw[1] * relevance_out[key] + gw[2] * importance_out[key])
        master_out[key] = total_score
        detailed_scores[key] = {
            'recency_score': recency_out[key],
            'relevance_score': relevance_out[key],
            'importance_score': importance_out[key],
            'total_score': total_score
        }

    master_out = top_highest_x_values(master_out, n_count)
    for key in master_out.keys():
        for node in nodes:
            if node['node_id'] == key:
                node['last_accessed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    retrieved = {
        'nodes': [
            {**node, 
             'scores': detailed_scores[node['node_id']]} 
            for node in nodes if node['node_id'] in master_out.keys()
        ]
    }
    return retrieved

def save_nodescore_to_file(related_nodes, filename="nodes_score.json"):
    with open(filename, "w") as file:
        json.dump(related_nodes, file, indent=4)






# # Load questions and answers for subsequent evaluation

def load_qa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Response prompt function
def build_response_prompt(question, related_nodes):
    context = "\n\n".join([f"Context {i+1}: {node['description']}" for i, node in enumerate(related_nodes)])
    prompt = f"""
    Based on the following context, answer the user's question in a clear and concise manner, Keep the response under the 300-word characters:

    {context}

    User's question: "{question}"
    """

    return prompt



def generate_contextual_response(question, embeddings, memory, deployment_name="gpt-4"):
    
    #memory = load_memory(memory_file)
    #embeddings = load_embeddings(embeddings_file)
    #intent = identify_intent(question)
    
    related_nodes = retrieve(question, embeddings, memory)
    

    related_nodes_info = []
    if related_nodes['nodes']:
        for node in related_nodes['nodes']:
            node_info = f"Node ID: {node['node_id']}, Description: {node['description']}"
            if 'filling' in node and node['filling']:  
                node_info += f", Filling: {node['filling']}"
            related_nodes_info.append(node_info)

        
        prompt = build_response_prompt(question, related_nodes['nodes'])

        response = client_chat.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        
        response_content = response.choices[0].message.content.strip()
    else:
        
        response_content = get_chat_response(deployment_name, [{"role": "user", "content": question}])

    
    return (response_content, related_nodes_info)


def process_questions_and_generate_responses(questions, embeddings, memory):
    responses = []  
    related_nodes_info_list = []  

    for question in questions:   
        #question = item['Q']
        #intent = identify_intent(question)
        
        related_nodes = retrieve(question, embeddings, memory)

        
        response_content, related_nodes_info = generate_contextual_response(question, embeddings, memory, "gpt-4")

        
        responses.append(response_content)

        
        related_nodes_info_list.append(related_nodes_info)

    
    return responses, related_nodes_info_list

# load responses_answers.json 
def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

# Generates a response and inserts it into the "hyp" field of responses_answers.json
def save_responses_to_file(questions, responses_file_path, embeddings, memory):
    responses_data = load_responses(responses_file_path)

    generated_responses, related_nodes_info_list = process_questions_and_generate_responses(questions, embeddings, memory)

    for i, response in enumerate(generated_responses):
        if i < len(responses_data):
            responses_data[i]["hyp"] = response

    with open(responses_file_path, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, ensure_ascii=False, indent=4)

    print("The generated response was successfully stored in the responses_answers.json file.")
# load related_nodes.json 
def load_related_nodes(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, create it 
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)  # Initialize with an empty list
            return []

# save related nodes to related_nodes.json
def save_rn_to_file(questions, related_nodes_file_path, embeddings, memory):
    
     #Initialize an empty list
    
    related_nodes_data = []

    # Process each question, retrieve related nodes
    _, related_nodes_info_list = process_questions_and_generate_responses(questions, embeddings, memory)
    
    # Add related nodes info to the related_nodes_data list
    for i, related_nodes_info in enumerate(related_nodes_info_list):
        if i < len(questions):  # Ensure we don't go out of index range
            related_nodes_data.append({
                "question": questions[i], 
                "related_nodes": related_nodes_info
            })

    # Save the updated related nodes data to the file
    with open(related_nodes_file_path, 'w', encoding='utf-8') as f:
        json.dump(related_nodes_data, f, ensure_ascii=False, indent=4)

    print("The related nodes were successfully stored in the related_nodes.json file.")


if __name__ == "__main__":

    memory_file = 'reflection_nodes.json'
    embeddings_file = 'reflection_embeddings.json'
    responses_file_path = 'responses_answers.json'
    related_nodes_file_path= 'related_nodes.json'
    memory = load_memory(memory_file)
    embeddings = load_embeddings(embeddings_file)
    deployment_name = "gpt-4"

    
    print("Loaded memory nodes count:", len(memory['nodes']))
    print("Loaded embeddings count:", len(embeddings))
    # Define your weights (Global Weights)
    #gw = [0.33, 0.33, 0.33]
    
    # Print the weights being used
    #print("Weights for retrieval - Recency:", gw[0], "Relevance:", gw[1], "Importance:", gw[2])
    

    qa_data_file = 'questions.json'
    questions = load_qa_data(qa_data_file)

    questions = [item['Q'] for item in questions]
    response, related_nodes_info_list = process_questions_and_generate_responses(questions, embeddings, memory) # Pass a list with single question

    save_responses_to_file(questions, responses_file_path, embeddings, memory)
    # Save the related nodes to file
    save_rn_to_file(questions, related_nodes_file_path, embeddings, memory)
    
    all_data = []

    for question in questions:
       #print(f"Question: {question}")
       related_nodes = retrieve(question, embeddings, memory)
       all_data.append({'Question': question, 'Nodes': related_nodes['nodes']})

    #print("Relevant Nodes:")
    #for node in related_nodes['nodes']:
        #scores = node['scores']
        #print(f"Node ID: {node['node_id']}, Scores: {scores}")
    #print("\n" + "="*50 + "\n")

    # 保存所有数据到文件
    with open("nodes_score.json", "w") as file:
      json.dump(all_data, file, indent=4)
