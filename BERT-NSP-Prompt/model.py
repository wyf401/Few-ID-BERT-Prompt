import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
from utils.activations import ACT2FN



class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)
        self.cls = BertPreTrainingHeads(config)


        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # print(seq_relationship_score)
        seq_relationship_score = seq_relationship_score[:,0]
        # seq_relationship_score = torch.softmax(seq_relationship_score,dim=-1)[:,0]
        # print(seq_relationship_score)
        # import sys
        # sys.exit()
        return seq_relationship_score
        # return intent_logits, slot_logits

    def get_mask_logits(self, prediction_scores, masked_index):
        mask_logits = []
        for batch_idx, index in enumerate(masked_index):
            single_logits = prediction_scores[batch_idx,index:index+1,:]
            mask_logits.append(single_logits)
        return torch.cat(mask_logits,dim=0)


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states