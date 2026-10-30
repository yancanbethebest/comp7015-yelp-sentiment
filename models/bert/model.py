"""BERT encoder with a two-label classification head (Hugging Face implementation)."""
import torch
from transformers import BertConfig, BertForSequenceClassification


def build_model(pretrained_name="google-bert/bert-base-uncased"):
    return BertForSequenceClassification.from_pretrained(pretrained_name, num_labels=2)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BertForSequenceClassification(BertConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
