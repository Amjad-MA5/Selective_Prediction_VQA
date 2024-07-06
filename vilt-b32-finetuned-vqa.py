from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import requests
from io import BytesIO

class ViltVQA:
    def __init__(self):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def predict(self, image_input, question):
        # Load image if image_input is a URL or file path
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
        else:
            # Assume image_input is already a PIL image object
            image = image_input

        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert("RGB")

        # Process the image and the question
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            answer = self.processor.tokenizer.decode(logits.argmax(-1))

        # Get confidence score
        confidence = torch.softmax(logits, dim=-1).max().item()
        return answer, confidence

def predict_with_selection(vqa_model, dataset, threshold=0.5):
    results = []
    for example in dataset:
        image_path = example["image"]
        question = example["question"]
        answer, confidence = vqa_model.predict(image_path, question)
        if confidence >= threshold:
            results.append({
                "image_id": example["image_id"],
                "question_id": example["question_id"],
                "question": question,
                "answer": answer,
                "confidence": confidence
            })
        else:
            results.append({
                "image_id": example["image_id"],
                "question_id": example["question_id"],
                "question": question,
                "answer": "Prediction confidence too low",
                "confidence": confidence
            })
    return results
