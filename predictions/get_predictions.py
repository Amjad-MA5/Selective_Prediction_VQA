'''
This code will provide logits and labels for calibration
'''
from transformers import ViltProcessor, ViltForQuestionAnswering
import json
import torch
import sys
sys.path.append("/teamspace/studios/this_studio/Selective_Prediction_VQA")

def get_model():
    """
        returns a finetuned ViLT model on VQAv2 dataset
    """
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return (model, processor)


def answers_to_labels(answers, config):
     """
    Converts answers to corresponding labels using the model's config.

    Args:
        answers (list): A list of answer annotations.
        config: The model configuration containing the label2id mapping.

    Returns:
        list: A list of labels corresponding to the answers.
    """
    
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

    """
    Saves the logits and labels to a file.

    Args:
        logits (list): A list of logits.
        labels (list): A list of labels.
        batch_no (int): The batch number to be used in the file name.

    The data is saved in a specified folder with the filename containing the batch number.
    """
    
    data = {}
    data['logits'] = logits
    data['labels'] = labels
    file_name = "Logits_and_labels"+ str(batch_no) +  ".pt"
    folder = "/teamspace/studios/this_studio/Selective_Prediction_VQA/predictions/logits_and_labels/"
    torch.save(data, folder+file_name)
    print("Saved data for batch no:", batch_no )

    

def prediction(model, processor, dataset, device ='cpu'):
    """
        Input: VQA model 
        dataset : dataset to generate logits 
        Output: logits and its corresponding correct answers annotated by humans
    """
    logits_accumulated = []
    labels_accumulated = []
    batch_size = 100
    batch_no = 0
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    for data in dataset:
        try:
            image = data['image']
            text = data['question']
            encoding = processor(image, text, return_tensors="pt")
            encoding.to(device)
            # forward pass
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                # idx = logits.argmax(-1).item()
                # print("Predicted answer:", model.config.id2label[idx])
                logits_accumulated.append(logits.cpu().detach().numpy())
            labels_accumulated.append(answers_to_labels(data['answers'], model.config))
            if len(labels_accumulated) >= batch_size:
                save_logits_n_label(logits_accumulated, labels_accumulated, batch_no)
                batch_no += 1
                logits_accumulated = []
                labels_accumulated = []
        except Exception as e:
            print(e)
    
    if len(labels_accumulated) != 0  :
        save_logits_n_label(logits_accumulated, labels_accumulated, batch_no)
    return

