'''
This code will provide logits and labels for calibration
'''
from transformers import ViltProcessor, ViltForQuestionAnswering
import json
import torch
import sys
sys.path.append("/teamspace/studios/this_studio/Selective_Prediction_PathVQA")

def get_model():
    """
        returns a finetuned ViLT model on VQAv2 dataset
    """
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return (model, processor)


def answers_to_labels(answers, config):
    labels = []
    AnyAnswer = False
    for annotation in answers:
        try:
            AnyAnswer = True
            labels.append(config.label2id[annotation['answer']])
        except:
            pass
    if not AnyAnswer:        
        print("Oops, empty label list")
    return labels
        
        
def save_logits_n_label(logits, labels, batch_no):
    data = {}
    data['logits'] = logits
    data['labels'] = labels
    file_name = "Logits_and_labels"+ str(batch_no) +  ".pt"
    folder = "/teamspace/studios/this_studio/Selective_Prediction_PathVQA/predictions/logits_and_labels/"
    torch.save(data, folder+file_name)
    print("Saved data for batch no:", batch_no )

    

def prediction(model, processor, dataset):
    """
        Input: VQA model 
        dataset : dataset to generate logits 
        Output: logits and its corresponding correct answers annotated by humans
    """
    logits_accumulated = []
    labels_accumulated = []
    batch_size = 100
    batch_no = 0
    for data in dataset:
        image = data['image']
        text = data['question']
        encoding = processor(image, text, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            # idx = logits.argmax(-1).item()
            # print("Predicted answer:", model.config.id2label[idx])
            logits_accumulated.append(logits.detach().numpy())
        labels_accumulated.append(answers_to_labels(data['answers'], model.config))
        if len(labels_accumulated) >= batch_size:
            save_logits_n_label(logits_accumulated, labels_accumulated, batch_no)
            batch_no += 1
            logits_accumulated = []
            labels_accumulated = []
    if len(labels_accumulated) != 0  :
        save_logits_n_label(logits_accumulated, labels_accumulated, batch_no)
    return

