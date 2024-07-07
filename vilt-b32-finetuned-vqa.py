from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np

class ViltVQA:
    def __init__(self):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def predict(self, images, questions):
        # Process the images and questions
        inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Calculate probabilities and get confidence scores
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        answer_idxs = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        answers = [self.processor.tokenizer.decode([idx]) for idx in answer_idxs]

        return answers, confidences, probs

def predict_with_selection(vqa_model, dataloader, threshold=0.5):
    results = []
    all_predictions = []
    for batch in dataloader:
        image_paths = batch["image"]
        questions = batch["question"]

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        answers, confidences, probs = vqa_model.predict(images, questions)
        all_predictions.append(probs)

        for i, (answer, confidence) in enumerate(zip(answers, confidences)):
            if confidence >= threshold:
                results.append({
                    "image_id": batch["image_id"][i],
                    "question_id": batch["question_id"][i],
                    "question": questions[i],
                    "answer": answer,
                    "confidence": confidence
                })
            else:
                results.append({
                    "image_id": batch["image_id"][i],
                    "question_id": batch["question_id"][i],
                    "question": questions[i],
                    "answer": "Prediction confidence too low",
                    "confidence": confidence
                })

    all_predictions = np.concatenate(all_predictions, axis=0)
    return results, all_predictions