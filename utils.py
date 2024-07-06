import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import matplotlib.pyplot as plt

def save_predictions(predictions, file_path):
    with open(file_path, "w") as f:
        json.dump(predictions, f, indent=4)

def load_ground_truths(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return {ann['question_id']: ann['answers'] for ann in annotations['annotations']}

def evaluate(results, ground_truths):
    correct = 0
    total = len(results)
    confident = 0
    correct_confident = 0

    for result in results:
        if result['prediction'] != 'Low confidence':
            confident += 1
            if result['prediction'] in ground_truths[result['question_id']]:
                correct_confident += 1
        if result['prediction'] in ground_truths[result['question_id']]:
            correct += 1

    overall_accuracy = correct / total
    confident_accuracy = correct_confident / confident if confident > 0 else 0
    coverage = confident / total

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Accuracy on Confident Predictions: {confident_accuracy:.4f}")
    print(f"Coverage: {coverage:.4f}")

    return overall_accuracy, confident_accuracy, coverage

def plot_results(overall_accuracy, confident_accuracy, coverage):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Overall Accuracy')
    plt.bar(['Overall'], [overall_accuracy])

    plt.subplot(1, 3, 2)
    plt.title('Confident Accuracy')
    plt.bar(['Confident'], [confident_accuracy])

    plt.subplot(1, 3, 3)
    plt.title('Coverage')
    plt.bar(['Coverage'], [coverage])

    plt.show()