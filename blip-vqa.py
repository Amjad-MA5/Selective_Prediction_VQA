from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import torch

def blip_vqa_prediction_with_probability_score(image_path, question):
    # Load a pre-trained VQA model and processor
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    image = Image.open(requests.get(image_path, stream=True).raw)
    encoding = processor(text= question, images=image, return_tensors="pt")

    outputs = model(**encoding)
    logits = outputs.logits

    probs = torch.softmax(logits, dim=1)
    max_prob, predicted_class = torch.max(probs, dim=1)

    predicted_answer = model.config.id2label[predicted_class.item()]

    return predicted_answer, max_prob.item()

image_path = "https://prod-images-static.radiopaedia.org/images/9289883/1c20962e46c92ee83a3f551adb24fa_big_gallery.jpg"
question = "Which part of the body is in the picture?"

answer, confidence = blip_vqa_prediction_with_probability_score(image_path, question)
print(f"Answer: {answer}, Confidence: {confidence:.2f}")



