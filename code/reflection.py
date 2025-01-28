
"""reflection.py

1. Process the user's conversation data set into memory nodes

2. Memory reflection

3. Update to the file
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
    # Extract the content of the first response
    return response.choices[0].message.content

# Make an Embedding Vector API request and return the embedding vector of the text
def get_embeddings(texts, model="text-embedding-ada-002"):
    '''Encapsulate OpenAI's Embedding model interface'''
    data = client_embedding.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

# Memory data loading function
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

# Generate high-level questions,
def generate_focal_points(memory, n=10, deployment_name="gpt-4"):
    recent_nodes = memory['nodes'][-10:] if len(memory['nodes']) > 10 else memory['nodes']
    prompt = f"Based on the following descriptions of recent patient interactions, generate {n} in-depth questions focusing on trends and changes in health indicators or habits. Each question should explore changes over time and how these changes might impact health. The design of the questions should consider the following aspects:\n"
    prompt += "\n".join([
    "Inquire about specific health issues that have arisen and the key health indicators of concern.",
    "Ask about the potential causes of these health issues.",
    "Frame questions in an inquiry format, such as: 'How has my exercise frequency been recently?'",
    "Inquire about the patient's psychological and emotional state.",
    "Ask about any specific health recommendations provided by the AI assistant.",
    "Inquire about recent conversation topics and related health discussions."
     ])

    for node in recent_nodes:
        prompt += f"{node['description']}\n"

    # print("Generated Prompt for Focal Points:", prompt)  # print Generated Prompt 

    response = client_chat.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant trained to analyze recent events and generate relevant questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5,
        stop=["\n11."]  # Stop after generating question 10
    )

    questions_text = response.choices[0].message.content.strip()
    questions = questions_text.split('\n') if questions_text else []
    #print("Generated Questions:", questions)  # print Generated Questions
    return questions

# Relevance function

def question_relevance(nodes, question_embedding, embeddings):
    """Calculate the relevance score of a node"""
    relevance_out = {}
    for node in nodes:
        try:
            node_embedding = embeddings[node['embedding_key']]
            relevance_out[node['node_id']] = cos_sim(node_embedding, question_embedding)
        except KeyError:
            # Handle the situation where no embedding is found,  log or use the default value
            print(f"Embedding not found for key: {node['embedding_key']}")
            relevance_out[node['node_id']] = 0  # Use default relevance score
    return relevance_out

def cos_sim(a, b):
    """Calculates the cosine similarity between two vectors"""
    a = np.squeeze(a)  # Make sure a is a one-dimensional array
    b = np.squeeze(b)  # Make sure b is a one-dimensional array
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve function
def retrieve_question_related_nodes(questions, embeddings, memory, n_count=10):
    """
    Retrieve historical conversation nodes related to the question and score them according to relevance.
    """

    # Initialize an empty dictionary retrieved
    retrieved = {}

    # Deal with every question
    for question in questions:
        if question.strip():  
            question_embedding = get_embeddings(question)  
            # Calculate the correlation with all nodes
            relevance_scores = question_relevance(memory['nodes'], question_embedding, embeddings)
            # Sort by relevance score and get the highest n_count nodes
            top_nodes_keys = sorted(relevance_scores, key=relevance_scores.get, reverse=True)[:n_count]
            # Add the most relevant nodes to the result dictionary
            retrieved[question] = [node for node in memory['nodes'] if node['node_id'] in top_nodes_keys]

    return retrieved

# Build insights
def build_insights_and_evidence_prompt(question, nodes):
    """
    Construct a prompt for insights and evidence based on the question and related node content, 
    and limit the response to 300 words or less.
    """
    prompt = f"Please answer in English: Based on the following details related to the question '{question}', extract insights and provide evidence. Please keep your response within 300 words.\n\n"
    for node in nodes:
        prompt += f"- {node['description']}\n"
    prompt += "\nPlease provide concise insights and supporting evidence."
    return prompt


def generate_insights_and_evidence(questions, retrieved_nodes, deployment_name="gpt-4"):
    """
    Generate insights and evidence based on the question and retrieved nodes.
    """
    insights_and_evidence = {}
    for question in questions:
        prompt = build_insights_and_evidence_prompt(question, retrieved_nodes[question])
        # Define the messages variable to provide context to the language model
        messages=[{"role": "system", "content": "You are a helpful assistant, keep answer under 300 words."},
                  {"role": "user", "content": prompt}]
        response = client_chat.chat.completions.create(
            model=deployment_name,
            messages=messages, # Use the defined messages variable
            max_tokens=1000,
            temperature=0.5
        )
        insights_and_evidence[question] = {
            "insights": response.choices[0].message.content.strip(),
        }
    return insights_and_evidence

memory_file = 'nodes.json'
embeddings_file = 'embeddings.json'
memory = load_memory(memory_file)
embeddings = load_embeddings(embeddings_file)
deployment_name = "gpt-4"
questions = generate_focal_points(memory, 10, deployment_name)


retrieved_nodes = retrieve_question_related_nodes(questions, embeddings, memory)
insights_and_evidence = generate_insights_and_evidence(questions, retrieved_nodes, deployment_name)



"""# Save the generated insights into memory nodes

"""

# Poignancy prompt function
def build_poignancy_prompt(description):
    prompt = f"""
    Based on the provided detailed description, please rate the poignancy (importance or impact) of the conversation on a scale from 1 to 10, where 1 represents the least important and 10 represents extremely important. When rating, please consider the following aspects:

    - **Changes in health indicators**: If health indicators show positive improvement, rate higher; if they show negative changes, rate higher.

    - **Treatment and prevention**: If the patient is actively pursuing treatment, rate higher; if the patient is not actively pursuing treatment, rate higher.

    - **Psychological state**: If the patient displays positive emotions, rate higher; if they display negative emotions, rate higher.

    - **Lifestyle habits**: If the patient's lifestyle habits improve, rate higher; if their lifestyle habits worsen, rate higher.

    Conversation content: "{description}"

    Poignancy score:
    """
    return prompt


#  Generate poignancy score function
def generate_poignancy_score(description):
    prompt = build_poignancy_prompt(description)
    messages = [
        {"role": "system", "content": "You are a medical assistant tasked with evaluating the emotional and clinical impact of patient conversations. Your goal is to rate the poignancy of the conversation on a scale of 1 to 10 based on the provided guidelines."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client_chat.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        score = int(float(response.choices[0].message.content.strip()))
        
        return min(max(score, 1), 10)
    except ValueError as e:
        
        print("Error parsing the model output:", e)
        
        return 1
    except Exception as e:
        
        print("An unexpected error occurred:", e)
        return 1


# Build spo prompt

def build_spo_prompt(description, deployment_name="gpt-4"):
    prompt = f"""
    First, please summarize this medical conversation in English in a single sentence. Focus on the key elements of the discussion between the assistant and the patient. Format the summary as follows: 'The assistant and (patient's name) discussed aspects of (disease name)', where you fill in the patient's name and the specific aspects of the disease discussed.

    {description}

    Second, extract the Subject, Predicate, and Object from the single sentence, and provide the result in the following format:
    Subject: <subject>
    Predicate: <predicate>
    Object: <object>
    """
    return prompt



# Use GPT to generate subject, predicate and object
def generate_spo(description, deployment_name="gpt-4"):
    prompt = build_spo_prompt(description, deployment_name)
    response = client_chat.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    response_text = response.choices[0].message.content.strip()

    spo = {"subject": "Unknown", "predicate": "unknown", "object": "unknown"}
    if "Subject:" in response_text and "Predicate:" in response_text and "Object:" in response_text:
        for line in response_text.split('\n'):
            if line.startswith("Subject:"):
                spo["subject"] = line.split(":", 1)[1].strip()
            elif line.startswith("Predicate:"):
                spo["predicate"] = line.split(":", 1)[1].strip()
            elif line.startswith("Object:"):
                spo["object"] = line.split(":", 1)[1].strip()
    else:
        print("Response format is incorrect or missing information.")

    return spo

# ConceptNode class: defines the structure of the node and various properties of the node
class ConceptNode:
    def __init__(self,
                 node_id: str,
                 node_count: int,
                 type_count: int,
                 node_type: str,
                 depth: int,
                 created,
                 expiration,
                 s: str,
                 p: str,
                 o: str,
                 description: str,
                 embedding_key: str,
                 poignancy: int,
                 keywords: set,
                 filling):
        """Initialize the ConceptNode object"""
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.type = node_type
        self.depth = depth
        self.created = created
        self.expiration = expiration
        self.last_accessed = created
        self.subject = s
        self.predicate = p
        self.object = o
        self.description = description
        self.embedding_key = embedding_key
        self.poignancy = poignancy
        self.keywords = keywords
        self.filling = filling

    def spo_summary(self):
        """Return the subject, predicate and object summary of a node"""
        return (self.subject, self.predicate, self.object)


def save_memory(memory, embeddings, memory_file, embeddings_file):
    """
    Save memory structure and embeddings to files.

    :param memory: dict, the memory structure containing nodes.
    :param embeddings: dict, the embeddings associated with nodes.
    :param memory_file: str, the filename for storing memory nodes.
    :param embeddings_file: str, the filename for storing embeddings.
    """
    # Save the memory nodes
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)
        print(f"Memory saved to {memory_file}")

    # Save the embeddings
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)
        print(f"Embeddings saved to {embeddings_file}")

def create_single_concept_node(memory, embeddings, question, insight, node_count, type_count, node_type="thought"):
    created_time = datetime.now()
    expiration_time = created_time + timedelta(days=10)

    # Questions and insights combined into node descriptions

    description = json.dumps({
        "question": question,
        "insights": insight
      }, ensure_ascii=False)



    spo = generate_spo(description)
    subject = spo["subject"]
    predicate = spo["predicate"]
    object = spo["object"]

    event_embedding = get_embeddings([description])
    embeddings[description] = event_embedding

    poignancy_score = generate_poignancy_score(description)

    new_node = {
        "node_id": str(node_count),
        "node_count": node_count,
        "type_count": type_count,
        "type": node_type,
        "depth": 1,
        "created": created_time.strftime('%Y-%m-%d %H:%M:%S'),
        "expiration": expiration_time.strftime('%Y-%m-%d %H:%M:%S'),
        "last_accessed": created_time.strftime('%Y-%m-%d %H:%M:%S'),
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "description": description,
        "embedding_key": description,
        "poignancy": poignancy_score,
        "filling": None  
    }
    memory['nodes'].append(new_node)
    return memory

def store_each_question_insight(memory, embeddings, questions, insights_and_evidence):
    node_count = len(memory['nodes'])  # Start from the existing number of nodes if adding to memory
    type_count = 0  # Assuming type_count isn't critical and can start from 0 for simplicity

    # Iterate over each question and its corresponding insights
    for question, insight in insights_and_evidence.items():
        # Create a memory node for each question and insight pair
        memory = create_single_concept_node(memory, embeddings, question, insight, node_count, type_count)

        # Increment counters for nodes and types
        node_count += 1
        type_count += 1  # Update this as needed if it has a specific meaning in your system

    # Save all changes after processing all questions and insights
    save_memory(memory, embeddings, 'reflection_nodes.json', 'reflection_embeddings.json')

    return memory

updated_memory = store_each_question_insight(memory, embeddings, questions, insights_and_evidence)

