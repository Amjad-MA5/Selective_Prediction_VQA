import torch
from torch.utils.data import DataLoader
from dataset.get_VQAv2_dataset import get_VQAv2_dataset
from risk_control import RiskControl
from vilt_b32_finetuned_vqa import ViltVQA, predict_with_selection
from utils import save_predictions, evaluate, plot_results, load_ground_truths, custom_collate_fn
import numpy as np

def main():
    # Load dataset
    dataset = get_VQAv2_dataset()
    val_dataset = dataset['validation']
    total_entries = len(val_dataset)
    val_dataset = val_dataset.select(range(total_entries - 1000, total_entries))

    # Create DataLoader for batch processing
    val_dataloader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=4, collate_fn=lambda x: custom_collate_fn(x, target_size=(512, 512)))

    # Load pre-trained model
    vqa_model = ViltVQA()

    # Predict on validation dataset with selection
    results, all_predictions = predict_with_selection(vqa_model, val_dataloader, threshold=0.2)

    # Calculate kappa and residuals
    kappa = np.max(all_predictions, axis=1)
    ground_truths = load_ground_truths(val_dataset)

    # Debugging: Check all_predictions shape and first few entries
    print(f"all_predictions shape: {all_predictions.shape}")
    #print(f"First few all_predictions: {all_predictions[:5]}")

    # Correct residual calculation and check first few entries
    residuals = np.array([1 if int(np.argmax(all_predictions[i])) not in ground_truths.get(val_dataset[i]['question_id'], []) else 0 for i in range(len(val_dataset))])
    #print(f"First few residuals: {residuals[:5]}")

    # Risk control parameters
    delta = 0.005
    rstar = 0.005

    # Calculate risk bound
    bound_calculator = RiskControl()
    theta, b_star = bound_calculator.bound(rstar, delta, kappa, residuals, split=True)

    # Apply thresholding based on risk control
    confident_predictions = kappa >= 0.5
    confident_results = [results[i] for i in range(len(results)) if confident_predictions[i]]

    # Debugging: Check first few confident_predictions
    #print(f"First few confident_predictions: {confident_predictions[:5]}")
    print(f"Number of confident results: {len(confident_results)}")

    # Evaluate confident predictions
    overall_accuracy, confident_accuracy, coverage = evaluate(confident_results, ground_truths)

    # Save and plot results
    save_predictions(confident_results, 'confident_predictions.json')
    plot_results(overall_accuracy, confident_accuracy, coverage, save_path='results_plot.png')

    print(f'Confident Prediction Accuracy: {confident_accuracy}')
    print(f'Theta: {theta}, Bound: {b_star}')

if __name__ == "__main__":
    main()
