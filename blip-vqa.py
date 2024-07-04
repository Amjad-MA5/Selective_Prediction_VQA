from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import torch

# Load a pre-trained VQA model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def blip_vqa(image_path, question, threshold=0.5):
    image = Image.open(requests.get(image_path, stream=True).raw)
    encoding = processor(text= question, images=image, return_tensors="pt")
    
