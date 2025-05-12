import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import os
def load_timesformer_model():
    """
    Load the pre-trained TimeSformer model for video classification.
    """
    # Load the processor and model from Hugging Face
    processor =AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 25)
    # Optionally load fine-tuned weights if available
    checkpoint_path = "/user/HS402/zs00774/Downloads/action-recognition-vit/timesformer_model.pth"  # Update this path if you have fine-tuned weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print("Loaded fine-tuned weights from:", checkpoint_path)
    else:
        print("Using pre-trained TimeSformer weights.")

    return processor, model
