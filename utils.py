import os
import json
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor, Resize
import numpy as np
from PIL import Image

def save_predictions(predictions, file_path):
    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError

    with open(file_path, "w") as f:
        json.dump(predictions, f, indent=4, default=convert)

def load_ground_truths(dataset):
    ground_truths = {}
    for entry in dataset:
        ground_truths[entry['question_id']] = [answer['answer'] for answer in entry['answers']]
    return ground_truths

def evaluate(predictions, ground_truths):
    correct = 0
    total = len(predictions)
    confident = 0
    correct_confident = 0

    if total == 0:
        print("No predictions to evaluate.")
        return 0, 0, 0

    missing_ground_truth_count = 0

    for result in predictions:
        question_id = result['question_id']
        answer = result['answer']
        print(f"Evaluating Question ID: {question_id}")

        if question_id not in ground_truths:
            # Attempt off-by-one correction
            if (question_id + 1) in ground_truths:
                question_id += 1
            elif (question_id - 1) in ground_truths:
                question_id -= 1
            else:
                print(f"Warning: Missing ground truth for question_id {question_id}")
                missing_ground_truth_count += 1
                continue

        ground_truth_answers = ground_truths[question_id]

        if answer != 'Prediction confidence too low':
            confident += 1
            if answer in ground_truth_answers:
                correct_confident += 1
                print(f"Confident and Correct: QID: {question_id}, Answer: {answer}, Ground Truths: {ground_truth_answers}")
            else:
                print(f"Confident but Incorrect: QID: {question_id}, Answer: {answer}, Ground Truths: {ground_truth_answers}")
        else:
            print(f"Low Confidence: QID: {question_id}, Answer: {answer}")

        if answer in ground_truth_answers:
            correct += 1
            print(f"Correct: QID: {question_id}, Answer: {answer}, Ground Truths: {ground_truth_answers}")
        else:
            print(f"Incorrect: QID: {question_id}, Answer: {answer}, Ground Truths: {ground_truth_answers}")

    overall_accuracy = correct / total
    confident_accuracy = correct_confident / confident if confident > 0 else 0
    coverage = confident / total

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Accuracy on Confident Predictions: {confident_accuracy:.4f}")
    print(f"Coverage: {coverage:.4f}")

    return overall_accuracy, confident_accuracy, coverage

def plot_results(overall_accuracy, confident_accuracy, coverage, save_path=None):
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

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def custom_collate_fn(batch, target_size=(512, 512)):
    images = []
    questions = []
    question_ids = []
    image_ids = []
    answers = []
    multiple_choice_answers = []
    question_types = []
    answer_types = []

    resize_transform = Resize(target_size)


    for entry in batch:
        image = entry['image']
        image = image.convert("RGB")
        if image.size != target_size:
            image = resize_transform(image)
        images.append(ToTensor()(image))
        questions.append(entry['question'])
        question_ids.append(entry['question_id'])
        image_ids.append(entry['image_id'])
        answers.append(entry['answers'])
        multiple_choice_answers.append(entry['multiple_choice_answer'])
        question_types.append(entry['question_type'])
        answer_types.append(entry['answer_type'])

    return {
        'images': torch.stack(images),
        'questions': questions,
        'question_ids': question_ids,
        'image_ids': image_ids,
        'answers': answers,
        'multiple_choice_answers': multiple_choice_answers,
        'question_types': question_types,
        'answer_types': answer_types,
    }
