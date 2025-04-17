import json
import os

# Loading the JSON files
def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

# Calculating Top-K Recall
def calculate_top_k_recall(related_nodes_file, true_nodes_file, k_list):
    # Load the data
    model_data = load_json(related_nodes_file)
    ground_truth_data = load_json(true_nodes_file)
    
    recall_results = {k: [] for k in k_list}  # Store recall for each K

    for model_entry, gt_entry in zip(model_data, ground_truth_data):
        retrieved_nodes = model_entry["related_nodes"]  # Model output as a list
        relevant_nodes = set(gt_entry["true_nodes"])  # Ground truth as a set
        
        for k in k_list:
            top_k_retrieved = set(retrieved_nodes[:k])  # Top-K results
            if len(relevant_nodes) == 0:  # Avoid division by zero
                recall = 1.0 if len(top_k_retrieved) == 0 else 0.0
            else:
                recall = len(relevant_nodes & top_k_retrieved) / len(relevant_nodes)
            recall_results[k].append(recall)
    
    # Calculate average recall for each K
    avg_recall = {k: sum(recall_results[k]) / len(recall_results[k]) for k in k_list}
    return recall_results, avg_recall

if __name__ == "__main__":
    # File paths
    related_nodes_file = 'related_nodes.json'
    true_nodes_file = 'true_nodes.json'
    # Output file path
    output_file = os.path.join (os.getcwd(),'top_k_recall_results.json')

    # Top-K values to evaluate
    k_values = [1, 2, 3, 5]

    # Calculate Top-K Recall
    recall_scores, average_recall = calculate_top_k_recall(related_nodes_file, true_nodes_file, k_values)

    # Prepare output data
    output_data = {
        "Per-question Recall Scores (Top-K)": recall_scores,
        "Average Recall Rate for each K": average_recall
    }

    # Save output to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {output_file}")
