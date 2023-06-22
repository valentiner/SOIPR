from transformers.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.crossattention = BertAttention(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask
    ):
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        attention_output = self.crossattention(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )[0]
        return attention_output


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class SOIRP(BertPreTrainedModel):
    def __init__(self, config):
        super(SOIRP, self).__init__(config)
        self.bert = BertModel(config=config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.Lr_e1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.Lr_e2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attention = BertAttention(config)
        self.elu = nn.ELU()
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.Cr1 = nn.Linear(config.hidden_size, config.en_label)
        self.Cr2 = nn.Linear(config.hidden_size, config.num_p * config.ht_label)
        self.rounds = config.rounds
        self.e_layer = DecoderLayer(config)
        self.crossattention = CrossAttention(config)
        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr2.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        embed = self.get_embed(token_ids, mask_token_ids)
        B, L = embed.shape[0], embed.shape[1]
        e1 = self.Lr_e1(embed)
        e2 = self.Lr_e2(embed)
        for i in range(self.rounds):
            if i != self.rounds - 1:
                e1_ = self.crossattention(e1, e2, mask_token_ids)
                e2_ = self.crossattention(e2, e1, mask_token_ids)
                e1 = e1_
                e2 = e2_
                h = self.elu(e1_.unsqueeze(2).repeat(1, 1, L, 1) * e2_.unsqueeze(1).repeat(1, L, 1, 1))
            if i == self.rounds - 1:
                h = e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1)
                en_table = self.elu1(h)
                ht_table = self.elu2(h)
                en_table = self.Cr1(en_table)
                ht_table = self.Cr2(ht_table)
            else:
                e1_ = h.max(dim=2).values
                e2_ = h.max(dim=1).values
                e1 = e1 + self.e_layer(e1_, embed, mask_token_ids)[0]
                e2 = e2 + self.e_layer(e2_, embed, mask_token_ids)[0]
        return en_table, ht_table.reshape([B, L, L, self.config.num_p, self.config.ht_label])

    def get_embed(self, token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed = bert_out[0]
        embed = self.dropout(embed)
        return embed
