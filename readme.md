# Long-Term Health Monitoring (LTHM) Dataset and Memory Framework

This repository contains the implementation of a memory-enhanced framework for processing longitudinal healthcare dialogues, as described in the paper "Leveraging the Long-Term Health Monitoring Dataset for Personalized Healthcare: A Large Language Model Approach".

## Overview

The LTHM framework is designed to address the limitations of Large Language Models (LLMs) in handling long-term memory for personalized healthcare applications. It implements a Memory Reflection and Dynamic Retrieval Weights (RTD) mechanism to enhance LLM performance in longitudinal patient interaction scenarios.

Key components:
- Memory storage system for healthcare dialogues
- Memory reflection for synthesizing trends and insights
- Dynamic retrieval mechanism with adjustable weights
- Response generation based on retrieved contextual information
- Evaluation metrics for memory retrieval and response quality

## Requirements

- Python 3.8+
- OpenAI API access (Azure OpenAI Service)
- Required packages:
  ```
  openai
  numpy
  rouge
  ```

## Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install openai numpy rouge
   ```
3. Configure your OpenAI API credentials in each script:
   - Replace `"your_api_key_here"` with your actual API key
   - Replace `"your_api_base_here"` with your actual API base URL

## File Structure

- `memorystorage.py`: Processes and stores healthcare dialogue data
- `reflection.py`: Generates and stores reflective insights from memory
- `response_reflection.py`: Generates responses based on retrieved memories
- `rouge_evaluation.py`: Evaluates response quality using ROUGE metrics
- `top_k_recall.py`: Calculates recall metrics for memory retrieval
- `run_all_reflection.py`: Orchestrates the execution of all scripts in sequence

## Dataset

The system uses a Long-Term Health Monitoring dataset containing patient-doctor dialogues. The dataset should be stored in `data.json` in the following format:

```json
{
  "YYYY-MM-DD": {
    "dialogue": {
      "Assistant": "...",
      "User": "..."
    }
  },
  ...
}
```

## Execution Steps

The system should be run in the following sequence:

1. **Memory Storage**: Processes raw dialogue data into structured memory nodes
2. **Memory Reflection**: Analyzes stored memories to generate insights
3. **Response Generation**: Retrieves relevant memories and generates responses
4. **Evaluation**: Evaluates the quality of generated responses

You can run the complete pipeline using:

```
python run_all_reflection.py
```

Alternatively, you can run each script individually in the correct sequence:

```
python memorystorage.py
python reflection.py
python response_reflection.py
python rouge_evaluation.py
```

## Script Details

### 1. memorystorage.py

**Purpose**: Process raw dialogue data into structured memory nodes

**Input**: 
- `data.json`: Raw healthcare dialogue data

**Output**:
- `nodes.json`: Structured memory nodes
- `embeddings.json`: Vector embeddings of memory node content

**Functionality**:
- Converts dialogues into memory nodes with metadata
- Generates semantic embeddings for each node
- Calculates importance (poignancy) scores
- Extracts subject-predicate-object triples
- Creates timestamp and expiration data

### 2. reflection.py

**Purpose**: Generate reflective insights by analyzing existing memory nodes

**Input**:
- `nodes.json`: Memory nodes from previous step
- `embeddings.json`: Vector embeddings of memory nodes

**Output**:
- `reflection_nodes.json`: Enhanced memory nodes with reflective insights
- `reflection_embeddings.json`: Updated embeddings including reflection nodes

**Functionality**:
- Generates focal point questions based on memory content
- Retrieves relevant nodes for each question
- Synthesizes insights from retrieved nodes
- Creates new memory nodes containing reflective insights
- Updates embeddings for enhanced retrieval

### 3. response_reflection.py

**Purpose**: Generate responses to user queries using memory retrieval

**Input**:
- `reflection_nodes.json`: Memory nodes with reflective insights
- `reflection_embeddings.json`: Embeddings including reflection content
- `questions.json`: User queries for testing
- `responses_answers.json`: Template for storing responses

**Output**:
- Updated `responses_answers.json`: Contains generated responses
- `related_nodes.json`: Records which nodes were retrieved for each query

**Functionality**:
- Retrieves relevant memories based on query context
- Dynamically adjusts retrieval weights based on query type
- Generates contextually appropriate responses
- Records which memory nodes contributed to each response

### 4. rouge_evaluation.py

**Purpose**: Evaluate response quality using ROUGE metrics

**Input**:
- `responses_answers.json`: Contains generated responses and reference answers

**Output**:
- `rouge_scores.json`: ROUGE evaluation metrics

**Functionality**:
- Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores
- Measures lexical overlap between generated and reference responses
- Provides quantitative evaluation of response quality

### 5. top_k_recall.py (Optional)

**Purpose**: Evaluate memory retrieval performance

**Input**:
- `related_nodes.json`: Nodes retrieved for each query
- `true_nodes.json`: Ground truth nodes that should be retrieved

**Output**:
- `top_k_recall_results.json`: Recall metrics at different K values

**Functionality**:
- Calculates Recall@K metrics (K=1,2,3,5)
- Evaluates effectiveness of memory retrieval mechanism
- Compares different weighting strategies

## Retrieval Mechanisms

The system supports several retrieval mechanisms:

- **RTE** (Retrieval with Equal Weights): Equal weighting of recency, relevance, and importance
- **RTD** (Retrieval with Dynamic Adjustment): Dynamically adjusted weights based on query type
- **RTR** (Retrieval with Relevance-Only): Relevance-based retrieval

## Evaluation Metrics

- **Response Correctness**: Measures accuracy of generated responses
- **Contextual Coherence**: Assesses natural flow and connection to context
- **ROUGE Scores**: Measures lexical overlap with reference answers
- **Recall@K**: Evaluates memory retrieval effectiveness

## Citation

If you use this code or dataset in your research, please cite:

```
@inproceedings{tian2025leveraging,
  title={Leveraging the Long-Term Health Monitoring Dataset for Personalized Healthcare: A Large Language Model Approach},
  author={Tian, Limin and Shen, Ma, Jiaxin},
  booktitle={Proceedings of the IEEE Engineering in Medicine and Biology Conference (EMBC)},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md] file for details.
