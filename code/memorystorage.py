"""# Memory storage and processing of long-term monitoring data sets"""
import os
import json
import random
import datetime
from datetime import datetime, timedelta
import numpy as np
import openai
from openai import AzureOpenAI

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

# Poignancy prompt function
def build_poignancy_prompt(description):
    prompt = f"""
    Based on the provided detailed description, please rate the poignancy (importance or impact) of the conversation on a scale from 1 to 10, where 1 represents the least important and 10 represents extremely important. When rating, please consider the following aspects:

    - **Changes in health indicators**: If health indicators show positive improvement (e.g., improved blood sugar levels), rate higher; if they show negative changes(e.g., deterioration in blood sugar or kidney function), rate higher.

    - **Treatment and prevention**: If the patient is actively pursuing treatment, rate higher; if the patient is not actively pursuing treatment, rate higher.

    - **Psychological state**: If the patient displays positive emotions, rate higher; if they display negative emotions, rate higher.

    - **Lifestyle habits**: If the patient's lifestyle habits are improving (e.g., better diet, regular exercise), rate higher; if their habits are worsening (e.g., poor diet, lack of exercise), rate higher.


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
    First, please summarize this medical conversation in English in a single sentence. Focus on the key elements of the discussion between the assistant and the patient. Format the summary as follows: 'The assistant discussed with (Name) about which aspects of (disease name)', where you fill in the patient's name and specific aspects of the disease discussed:

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

    spo = {"subject": "unknown", "predicate": "unknown", "object": "unknown"}
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
        
        return (self.subject, self.predicate, self.object)


# Generate summary
def generate_summary(full_description, date_str, deployment_name="gpt-4"):

    # Extract and format the date from the provided date string
    formatted_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')

    # Ensure the description starts with the date
    full_description_with_date = f"Date: {formatted_date}\n\n{full_description}"

    # Build more instructive prompts, requiring the model to accurately extract key details for medical monitoring
    prompt = f"""
    Summarize the conversation from {formatted_date}, focusing on essential details related to medical monitoring. The summary should include:

    - Key monitoring indicators and their specific values (e.g., blood pressure, glucose levels).
    - Symptoms mentioned and their severity or frequency.
    - Details of any dietary or exercise recommendations, including specific foods or routines, and their prescribed durations.
    - Descriptions of any activities discussed, including their type, intensity, and duration.
    - Any related feelings, emotional states, or side effects expressed.
    - Medication advice given, including dosages and timing.

    Ensure that the summary accurately reflects the key dates and mentions of time durations such as 'one hour' in the conversation:

    Conversation Content: "{full_description_with_date}"
    """

    response = client_chat.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,  
        temperature=0.5,  
    )
    summary = response.choices[0].message.content.strip()
    return summary



def create_concept_node(description, node_count, type_count, memory, embeddings, fill, date_str):

    created_time = datetime.datetime.strptime(date_str, '%Y-%m-%d') 

    # Add random hours, minutes and seconds
    hours = random.randint(0, 12)  
    minutes = random.randint(0, 59)  
    seconds = random.randint(0, 59)  
    created_time = created_time.replace(hour=hours, minute=minutes, second=seconds)

    expiration_time = created_time + datetime.timedelta(days=30)

    summary = generate_summary(description, date_str)

    spo = generate_spo(description)
    subject = spo["subject"]
    predicate = spo["predicate"]
    object = spo["object"]

    event_embedding = get_embeddings([summary])
    embeddings[summary] = event_embedding

    poignancy_score = generate_poignancy_score(description)

    new_node = {
        "node_id": str(node_count),
        "node_count": node_count,
        "type_count": type_count,
        "type": "chat",
        "depth": 1,
        "created": created_time.strftime('%Y-%m-%d %H:%M:%S'),
        "expiration": expiration_time.strftime('%Y-%m-%d %H:%M:%S'),
        "last_accessed": created_time.strftime('%Y-%m-%d %H:%M:%S'),
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "description": summary,  
        "embedding_key": summary,
        "poignancy": poignancy_score,
        "keywords": list({subject, predicate, object}),
        "filling": fill  #Store the original conversation content
    }
    memory['nodes'].append(new_node)
    return memory

import datetime
def process_json_data(data):
    memory = {'nodes': []}  # Initialize memory structure to store nodes
    embeddings = {}  # Initialize embeddings dictionary
    node_count = 0
    type_count = 0

    # Iterate through each date in the data
    for date, conversations in data.items():
        # Initialize a string to accumulate all conversations for this date
        full_description = ""

        # Iterate through each conversation group under this date
        for conversation_group in conversations:
            # Each conversation group is a dictionary with an index as the key
            for idx, dialogue in conversation_group.items():
                # Accumulate the full conversation content into a single string
                full_description += f"Assistant: {dialogue['Assistant']} User: {dialogue['User']} "

        # Use the accumulated conversation content as "fill"
        fill = full_description.strip()  # Strip any trailing whitespace

        # Create a node using the accumulated conversations for this date
        memory = create_concept_node(full_description, node_count, type_count, memory, embeddings, fill, date)

        # Increment node count after processing each date
        node_count += 1
        # Increment type count
        type_count += 1

    # Return the updated memory structure and embeddings
    return memory, embeddings


# Read the file containing the conversation data
current_dir = os.path.dirname(os.path.abspath(__file__))

data_file = os.path.join(current_dir, 'data.json')

with open(data_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


# Process data and create nodes
memory, embeddings = process_json_data(data)

# Save the processed data to a JSON file
output_nodes_path = os.path.join(current_dir, 'nodes.json')
output_embeddings_path = os.path.join(current_dir, 'embeddings.json')

with open(output_nodes_path, 'w', encoding='utf-8') as f:
    json.dump(memory, f, ensure_ascii=False, indent=4)
with open(output_embeddings_path, 'w', encoding='utf-8') as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=4)