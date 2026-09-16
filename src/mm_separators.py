import torch
import torch.nn as nn
import math

def graph_rank_separators(
    self, alpha, theta, layer,
    features, position_ids, attention_mask
):
    _position_ids = position_ids
    _attention_mask = attention_mask

    if position_ids is None:
        position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
    
    if getattr(self.config, 'tokenizer_padding_side', 'right') != "right":
        raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")
        
    batch_size = features.shape[0]
    _device = features.device
    _dtype = features.dtype

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, features.shape[1]), dtype=torch.bool, device=features.device)
    else:
        attention_mask = attention_mask.bool()

    keep_length = [int(cur_image_token / math.e) for cur_image_token in self.image_tokens]
    rank_length = [int(len * 0.5) for len in keep_length]   # separators, directly obtained from features by attention score ranking

    # obtain query_states and key_states to calculate attention map
    hidden_states=features.clone()
    self_attn = self.layers[layer].self_attn
    hidden_states = self.layers[layer].input_layernorm(hidden_states)

    num_heads = self_attn.num_heads
    num_key_value_heads = self_attn.num_key_value_heads
    head_dim = self_attn.head_dim

    bsz, q_len, _ = hidden_states.size()

    query_states = self_attn.q_proj(hidden_states) # last text token
    key_states = self_attn.k_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos, sin = self_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # attention_mask
    if self.attention_type != 'flash_attention_2':
        raise NotImplementedError(f"`graph_rank_pruner` is only implemented for 'flash_attention_2' attention currently.")
    else:
        if attention_mask is not None:
            attention_mask = attention_mask[..., :q_len]

    features_list = []

    for i in range(batch_size):
        image_index= self.image_token_posi[i]
        if image_index == -1:
            cur_input_embeds = features[i]
            features_list.append(cur_input_embeds)
            attention_mask_list.append(attention_mask[i])
            continue
        
        # --------- 1) compute attention scores: last valid text token -> visual tokens ----------
        cur_key_states = key_states[i][:, image_index:image_index+self.image_tokens[i], :]
        cur_query_states = query_states[i]

        if self.training:
            pass
        else:
            text_query_states = cur_query_states[:,-1,:].unsqueeze(1)  # (num_head, 1, head_dim)

        attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim) #(num_head, text_token, seq_len)
        # attn_weights = attn_weights + text_attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) #(num_head, text_token, visual_len)
        attention_avg_text = attn_weights.mean(dim=0).squeeze(0) # ave across heads

        order = torch.argsort(attention_avg_text, descending=True)
        # choose top-k visual tokens ids
        num_visual_sep = rank_length[i]
        topk_visual_idx = order[:num_visual_sep]
        topk_visual_idx = topk_visual_idx.sort()[0]  # sort by id, ascending order, this is relative position in the visual token sequence
        selected_index = topk_visual_idx + image_index  # selected visual token positions in the original sequence

        # --------- 6) absolute positions in next layer ----------
        self.visual_sep_pos=selected_index  # update

        start_index = image_index + self.image_tokens[i]
        new_input_embeds = torch.cat([features[i][:image_index,:], features[i][selected_index,:], features[i][start_index:,:]], dim=0)
        features_list.append(new_input_embeds)

    max_len = max(embed.shape[0] for embed in features_list)

    ## update position_ids and attention_mask
    embeds_padded = []
    attention_mask_list = []
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=_device)
    for i, cur_new_embed in enumerate(features_list):
        cur_len = cur_new_embed.shape[0]
        dif = max_len - cur_len

        cur_new_embed = torch.cat([cur_new_embed, torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)], dim=0)    # pad to max_len
        embeds_padded.append(cur_new_embed)
        
        new_attention_mask = torch.cat(
            [   attention_mask[i][:self.image_token_posi[i]], # before image tokens
                attention_mask[i][self.image_token_posi[i]:self.image_token_posi[i]+rank_length[i]], # kept image tokens
                attention_mask[i][self.image_token_posi[i]+self.image_tokens[i]:]   # after image tokens
            ], dim=0)
        attention_mask_list.append(new_attention_mask)
        cur_len = new_attention_mask.sum().item()
        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)  # the position ids before image token merging
        self.image_tokens[i] = rank_length[i] # update image token number

    final_features = torch.stack(embeds_padded, dim=0).to(_dtype)  # (batch_size, max_len, hidden_size)
    new_attention_mask = torch.stack(attention_mask_list, dim=0)  # (batch_size, max_len)
    if _position_ids is None:
        position_ids = None
    if _attention_mask is None:
        new_attention_mask = None
    else:
        new_attention_mask = new_attention_mask.to(_attention_mask.dtype)

    return position_ids, new_attention_mask, final_features