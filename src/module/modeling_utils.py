from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsSkim(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None


@dataclass
class SequenceClassifierOutputSkim(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class QuestionAnsweringModelOutputSkim(QuestionAnsweringModelOutput):
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskedLMOutputSkim(MaskedLMOutput):
    attention_mask: Optional[torch.FloatTensor] = None
    skim_mask: Optional[torch.FloatTensor] = None
    skim_loss: Optional[torch.FloatTensor] = None
    classification_loss: Optional[torch.FloatTensor] = None
    tokens_remained: Optional[torch.FloatTensor] = None
    layer_tokens_remained: Optional[Tuple[torch.FloatTensor]] = None


def convert_softmax_mask_to_digit(skim_mask):
    # skim_mask [batch, from, to, seq_len]
    return (skim_mask == 0).to(dtype=torch.int64).unsqueeze(1).unsqueeze(1)

