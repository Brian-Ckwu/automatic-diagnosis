"""
    The models are adapted from Huggingface's BERT implementation.
"""

from argparse import Namespace
from packaging import version

import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutput

class BertPolEmbed(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.polarity_embeddings = nn.Embedding(
            num_embeddings=3, # NOTE: "not medical entity", "positive entity", and "negative entity"
            embedding_dim=config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )
    
    def forward(
        self,
        input_ids=None,
        token_pol_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None, 
        past_key_values_length=0
    ):
        # some useful variables
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # word embeddings
        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        # polarity embeddings
        if token_pol_ids is not None:
            pol_embeds = self.polarity_embeddings(token_pol_ids)
        else:
            pol_embeds = torch.zeros(size=inputs_embeds.size(), device=inputs_embeds.device)

        # token type embeddings
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)        
        type_embeds = self.token_type_embeddings(token_type_ids)

        # position embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        if self.position_embedding_type == "absolute":
            position_embeds = self.position_embeddings(position_ids)

        # whole embeddings
        embeddings = inputs_embeds + pol_embeds + type_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertPolModel(BertPreTrainedModel):

    def __init__(self, config, embeddings: BertPolEmbed, encoder):
        super().__init__(config)
        self.config = config

        self.embeddings = embeddings
        self.encoder = encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_pol_ids=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,        
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # TODO: check that the shape of token_pol_ids is the same as input_ids
        assert (token_pol_ids is not None) and (token_pol_ids.shape == input_ids.shape)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_pol_ids=token_pol_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def load_pretrained_weights(
        self, 
        bert_model: BertModel, 
        pol_embeds_init: str,
        bert_tokenizer: BertTokenizer = None,
        pos_token: str = None, 
        neg_token: str = None
    ):
        # Load embeddings
        # word embeddings
        self.embeddings.word_embeddings.load_state_dict(bert_model.embeddings.word_embeddings.state_dict())
        # polarity embeddings (0: empty embeddings / 1: positive embeddings / 2: negative embeddings)
        if pol_embeds_init == "random":
            self.embeddings.polarity_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        else:
            pol_tokens = bert_tokenizer.convert_tokens_to_ids([pos_token, neg_token])
            pol_tokens_tensor = torch.tensor(pol_tokens)
            pol_embeds = torch.concat(tensors=[torch.zeros((1, self.config.hidden_size)), bert_model.embeddings.word_embeddings(pol_tokens_tensor)])
            self.embeddings.polarity_embeddings = nn.Embedding.from_pretrained(pol_embeds, freeze=False) # TODO: exp with freeze?
        # type embeddings (in accordance with the original BERT model)
        self.embeddings.token_type_embeddings.load_state_dict(bert_model.embeddings.token_type_embeddings.state_dict())
        # position embeddings
        self.embeddings.position_embeddings.load_state_dict(bert_model.embeddings.position_embeddings.state_dict())

        # Load encoder
        self.encoder.load_state_dict(bert_model.encoder.state_dict())

class BERTPatient(nn.Module):

    def __init__(self, args: Namespace):
        self.encoder = load_bert_model(name=args.encoder_name, pol=args.add_pol_embeds)
        self.fc = nn.Linear(self.encoder.embeddings.word_embeddings.embedding_dim, 2) # positive & negative
    
    def forward(self, x):
        h_cls = self.encoder(**x).last_hidden_state[:, 0, :]
        logits = self.fc(h_cls)
        return logits # raw scores

    def calc_loss(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        return nn.functional.cross_entropy(logits, labels)

# Utility functions
def load_bert_model(name: str, pol: bool) -> nn.Module:
    bert_model = AutoModel.from_pretrained(name)
    
    if pol:
        bert_config = AutoConfig.from_pretrained(name)

        bert_pol_embeddings = BertPolEmbed(config=bert_config)
        bert_pol_model = BertPolModel(
            config=bert_config,
            embeddings=bert_pol_embeddings,
            encoder=bert_model.encoder
        )
        bert_pol_model.load_pretrained_weights(
            bert_model=bert_model,
            pol_embeds_init="random"
        )

        model = bert_pol_model
    else:
        model = bert_model

    return model