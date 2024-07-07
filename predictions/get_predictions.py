'''
This code will provide logits and labels for calibration
'''
from transformers import ViltProcessor, ViltForQuestionAnswering

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
        
        
    

def prediction(model, processor, dataset):
    """
        Input: VQA model 
        dataset : dataset to generate logits 
        Output: logits and its corresponding correct answers annotated by humans
    """
    logits_accumulated = []
    labels_accumulated = []
    for data in dataset:
        image = data['image']
        text = data['question']
        encoding = processor(image, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        # print("Predicted answer:", model.config.id2label[idx])
        logits_accumulated.append(logits)
        
        labels_accumulated.append(answers_to_labels(data['answers'], model.config))
    return logits_accumulated, labels_accumulated

def save_logits_n_label(logits, labels):
    data = {}
    data['logits'] = logits
    data['labels'] = labels
    with open("Logits_and_labels.json", "w") as f:
        json.dump(data, f)
