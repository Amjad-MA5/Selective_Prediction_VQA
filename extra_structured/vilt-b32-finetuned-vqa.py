from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import numpy as np

class ViltVQA:
    def __init__(self):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        self.processor.feature_extractor.do_rescale = False

    def predict(self, images, questions):
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # Debugging: Log the shape and some sample logits
        print(f"Logits shape: {logits.shape}")
        #print(f"Sample logits: {logits[:5]}")
        #print(f"Sample probabilities: {probabilities[:5]}")

        return probabilities.detach().cpu().numpy()

def predict_with_selection(vqa_model, dataloader, threshold=0.2):
    all_predictions = []
    results = []

    for batch in dataloader:
        images = batch['images'].to('cuda')
        questions = batch['questions']
        question_ids = batch['question_ids']

        probabilities = vqa_model.predict(images, questions)
        all_predictions.append(probabilities)

        for i in range(len(probabilities)):
            max_prob = np.max(probabilities[i])
            answer_index = np.argmax(probabilities[i])
            answer = vqa_model.model.config.id2label[answer_index]

            if max_prob < threshold:
                answer = 'Prediction confidence too low'
            
            results.append({
                    "image_id": batch["image_ids"][i],
                    "question_id": batch["question_ids"][i],
                    "question": questions[i],
                    "answer": answer,
                    "confidence": max_prob
            })
            print(f"Question ID: {question_ids[i]}, Confidence: {max_prob}")

    all_predictions = np.concatenate(all_predictions, axis=0)
    print(f"All predictions shape: {all_predictions.shape}")

    # Debugging: Print the number of confident results
    num_confident_results = sum(1 for result in results if result['confidence'] >= threshold)
    print(f"Number of confident results: {num_confident_results}")


    return results, all_predictions
