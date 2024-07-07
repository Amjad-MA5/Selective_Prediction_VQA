import torch
from torch.utils.data import DataLoader
from dataset.get_VQAv2_dataset import get_VQAv2_dataset
from risk_control import RiskControl
from vilt_b32_finetuned_vqa import ViltVQA, predict_with_selection
from utils import save_predictions, evaluate, plot_results, load_ground_truths

def main():
    # Load dataset
    dataset = get_VQAv2_dataset()
    val_dataset = dataset['testdev']

    # Create DataLoader for batch processing
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load pre-trained model
    vqa_model = ViltVQA()

    # Predict on validation dataset with selection
    results, all_predictions = predict_with_selection(vqa_model, val_dataloader)

    # Calculate kappa and residuals
    kappa = np.max(all_predictions, axis=1)
    ground_truths = load_ground_truths('/Selective_Prediction_PathVQA/dataset/data/VQAv2.hf/val/annotations.json')  # Adjust path as necessary
    residuals = (np.argmax(all_predictions, axis=1) != np.argmax(ground_truths, axis=1))

    # Risk control parameters
    delta = 0.05
    rstar = 0.05

    # Calculate risk bound
    bound_calculator = RiskControl()
    theta, b_star = bound_calculator.bound(rstar, delta, kappa, residuals, split=True)

    # Apply thresholding based on risk control
    confident_predictions = kappa >= theta
    confident_results = [result for i, result in enumerate(results) if confident_predictions[i]]

    # Evaluate confident predictions
    confident_predicted_x = all_predictions[confident_predictions]
    confident_ground_truths = ground_truths[confident_predictions]
    overall_accuracy, confident_accuracy, coverage = evaluate(confident_predicted_x, confident_ground_truths)

    # Save and plot results
    save_predictions(confident_results, 'confident_predictions.json')
    plot_results(overall_accuracy, confident_accuracy, coverage)

    print(f'Confident Prediction Accuracy: {confident_accuracy}')
    print(f'Theta: {theta}, Bound: {b_star}')

if __name__ == "__main__":
    main()