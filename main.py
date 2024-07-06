import torch
from dataset.get_VQAv2_dataset import get_VQAv2_dataset
from risk_control import RiskControl
from vilt_b32_finetuned_vqa import ViltVQA, predict_with_selection
from utils import save_predictions, evaluate, plot_results, load_ground_truths

def main():
    dataset = get_VQAv2_dataset()
    val_dataset = dataset['validation']
    print("This is test to see if this works dont be alarmed 1.1")
    vqa_model = ViltVQA()

    threshold = 0.5
    print("This is test to see if this works dont be alarmed 1.2")
    results = predict_with_selection( vqa_model, val_dataset, threshold)
    print("This is test to see if this works dont be alarmed 1.3")
    save_predictions(results, 'predictions.json')

    ground_truths = load_ground_truths('/Selective_Prediction_PathVQA/dataset/data/VQAv2.hf/val/annotations.json')  # Adjust path as necessary
    overall_accuracy, confident_accuracy, coverage = evaluate(results, ground_truths)
    print("This is test to see if this works dont be alarmed 1.4")
    plot_results(overall_accuracy, confident_accuracy, coverage)

if __name__ == "__main__":
    main()