import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import os
from transformers import TimesformerForVideoClassification
def load_timesformer_model():
    """
    Load the pre-trained TimeSformer model for video classification.
    
    """
    label_index_dict={'brush_hair': 0, 'cartwheel': 1, 'catch': 2, 'chew': 3, 'climb': 4, 'climb_stairs': 5, 'draw_sword': 6, 'eat': 7, 'fencing': 8, 'flic_flac': 9, 'golf': 10, 'handstand': 11, 'kiss': 12, 'pick': 13, 'pour': 14, 'pullup': 15, 'pushup': 16, 'ride_bike': 17, 'shoot_bow': 18, 'shoot_gun': 19, 'situp': 20, 'smile': 21, 'smoke': 22, 'throw': 23, 'wave': 24}
    index_label_dict={0: 'brush_hair', 1: 'cartwheel', 2: 'catch', 3: 'chew', 4: 'climb', 5: 'climb_stairs', 6: 'draw_sword', 7: 'eat', 8: 'fencing', 9: 'flic_flac',10: 'golf', 11: 'handstand', 12: 'kiss', 13: 'pick', 14: 'pour', 15: 'pullup', 16: 'pushup', 17: 'ride_bike', 18: 'shoot_bow', 19: 'shoot_gun', 20: 'situp', 21: 'smile', 22: 'smoke', 23: 'throw', 24: 'wave'}
    # Load the processor and model from Hugging Face
    processor =AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    ckpt = "facebook/timesformer-base-finetuned-k400"
    model = TimesformerForVideoClassification.from_pretrained(ckpt,label2id = label_index_dict,id2label = index_label_dict,ignore_mismatched_sizes = True)
    #model.classifier = torch.nn.Linear(model.config.hidden_size, 25)
    # Optionally load fine-tuned weights if available
    checkpoint_path = "/user/HS402/zs00774/Downloads/action-recognition-vit/timesformer_model.pth"  # Update this path if you have fine-tuned weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print("Loaded fine-tuned weights from:", checkpoint_path)
    else:
        print("Using pre-trained TimeSformer weights.")

    return processor, model
