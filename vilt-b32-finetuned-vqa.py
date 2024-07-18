from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import torch

def vilt_vqa_prediction_with_probability_score(image_path, question):

 """
    Generates a prediction and confidence score for visual question answering (VQA) using a pre-trained ViLT model.

    Args:
        image_path (str): URL of the image to be analyzed.
        question (str): The question related to the image.

    Returns:
        tuple: A tuple containing the predicted answer (str) and the confidence score (float).
    """


    
    # Load a pre-trained VQA model and processor
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # Fetch and open the image from the provided URL
    image = Image.open(requests.get(image_path, stream=True).raw)
    # Process the image and question
    encoding = processor(text= question, images=image, return_tensors="pt")

    # Get the model outputs
    outputs = model(**encoding)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    # Get the highest probability and the corresponding class
    max_prob, predicted_class = torch.max(probs, dim=1)
    
    # Convert the class ID to the actual answer label
    predicted_answer = model.config.id2label[predicted_class.item()]

    return predicted_answer, max_prob.item()

image_path = "https://prod-images-static.radiopaedia.org/images/9289883/1c20962e46c92ee83a3f551adb24fa_big_gallery.jpg"
question = "Which part of the body is in the picture?"

answer, confidence = vilt_vqa_prediction_with_probability_score(image_path, question)
print(f"Answer: {answer}, Confidence: {confidence:.2f}")



