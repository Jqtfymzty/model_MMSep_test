from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache


class MMSepCache(Cache):
    """
    A cache as described in the [MMSep paper - KDD 2026](https://arxiv.org/abs/2412.12094).
    Derived from Cache class in HuggingFace Transformers and MMSepCache in SepLLM (ICML 2025).
    """    
    @staticmethod
    def slice_on_1d(x, start, end):
        return x[:, start:end, ...]
    @staticmethod
    def slice_on_2d(x, start, end):
        return x[:, :, start:end, ...]
    @staticmethod
    def slice_on_3d(x, start, end):
        return x[:, :, :, start:end, ...]

    @staticmethod
    def sep_1bat_select_on_1d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 1. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, sep_index, ...]   # # batch x seqlen x head x dim  -->  sep_num x head x dim  
            padding_num = max_sep_num -  new_x_sep.shape[0]
            if padding_num > 0 :
                assert padding_num <= x.shape[1], f"`padding_num` should be <= `x.shape[1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, -padding_num: , ...]    #  padding_num x head x dim     
                return torch.cat([new_x_sep, new_x_pad ] , dim=0) # max_sep_num x head x dim 
            else:
                return new_x_sep #  max_sep_num x head x dim 

        if min_sep_num is None:
            return x[Bid, sep_index, ...]  # # batch x seqlen x head x dim -->  sep_num x head x dim    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, sep_index, ...]   # # batch x seqlen x head x dim -->  sep_num x head x dim               
            return new_x[ :min_sep_num, ...] # #  min_sep_num x head x dim      


    @staticmethod
    def sep_1bat_select_on_2d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 2. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, :, sep_index, ...]   # # batch x head x seqlen x dim -->  head x sep_num x dim  
            padding_num = max_sep_num -  new_x_sep.shape[-2]
            if padding_num > 0 :
                assert padding_num<= x.shape[-2], f"`padding_num` should be <= `x.shape[-2]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, -padding_num: , ...]    # head x padding_num x dim     
                return torch.cat([new_x_sep, new_x_pad ] , dim=-2) # head x max_sep_num x dim 
            else:
                return new_x_sep # head x max_sep_num x dim 

        if min_sep_num is None:
            return x[Bid, :, sep_index, ...]  # # batch x head x seqlen x dim -->  head x sep_num x dim    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, :, sep_index, ...]   # # batch x head x seqlen x dim -->  head x sep_num x dim            
            return new_x[:, :min_sep_num, ...] # #  head x min_sep_num x dim      


    @staticmethod
    def sep_1bat_select_on_3d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):    
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 3. 
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`); 
           otherwise, truncate to the shortest length (i.e. `min_sep_num`). 
        """        
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH: ## Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep =  x[Bid, :, :, sep_index, ...]   # # batch x head x dim x seqlen  -->  head x dim x sep_num 
            padding_num = max_sep_num -  new_x_sep.shape[-1]
            if padding_num > 0 :
                assert padding_num <= x.shape[-1], f"`padding_num` should be <= `x.shape[-1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, :, -padding_num:, ...]    # head x dim x padding_num     
                return torch.cat([new_x_sep, new_x_pad] , dim=-1) # head x dim x max_sep_num 
            else:
                return new_x_sep # head x dim x max_sep_num 

        if min_sep_num is None:
            return x[Bid, :, :, sep_index, ...]  # # batch x head x dim x seqlen -->  head x dim x sep_num    
        else: ## `min_sep_num` is provided. Need truncation
            new_x =  x[Bid, :, :, sep_index, ...]   # # batch x head x dim x seqlen -->  head x dim x sep_num          
            return new_x[:, :, :min_sep_num, ...] # #  head x dim x min_sep_num       

    DIM_TO_SLICE = {
        1: slice_on_1d,
        2: slice_on_2d,
        3: slice_on_3d,
    }
    
    BAT_DIM_TO_SELECT = {
        1: sep_1bat_select_on_1d,
        2: sep_1bat_select_on_2d,
        3: sep_1bat_select_on_3d,
    }

    def __init__(self,                               
                init_cache_size: Union[int, List] = 4,        
                sep_cache_size: Union[int, List] = 64,
                local_size: Union[int, List]=256, 
                cache_size: Union[int, List]=512,
                image_token_length: Union[int, List]=576,   # for llava v1.5
                image_start_pos: List[int] = None,  # for llava v1.5
                mmsep_layer: Optional[int] = None,
                SEP_ACCUMULATION: bool = True,
                USE_MAX_SEP_CACHE: bool = False,
                SEP_PADDING_IN_BATCH: bool = False,
                separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided.
                PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

                ## For inheritance & initialization states
                past_tok_ids: List[torch.Tensor] = None,  ## It saves all the token ids corresponding to the saved KVs for all layers in MMSepCache.                
                key_cache: List[torch.Tensor] = None,          
                value_cache: List[torch.Tensor] = None,

                ## For debugging
                PRINT_KV_RATIO_INSIDE: bool = False,
                print_KV_inside_per_steps: int = 1000,   
                _seen_tokens: int = 0, 
                _kept_kv_ratio: List[Tuple[int]] = None,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT: bool = False,
                APPLY_PES_INSIDE: bool = True,
                _shifted_position_ids:  List[torch.Tensor] = None,
                _rope_unsqueeze_dim: int = 1, ## The unsqueeze_dim when applying RoPE.
                _rope_seq_dim: int=1, ## The seq_len dimension for the `cos` or `sin` tensors.
                pe_scaling_factor:float = 1.0,
                pe_dim:int=128, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                max_position_embeddings: int = 8192, 
                base: int=10000,  ## The base for RoPE.               
                
                ## For basic transformer architecture
                k_seq_dim: int=2, ## The dimension for seq_len in key tensors
                v_seq_dim: int=2, ## The dimension for seq_len in value tensors
                layer_num: int = None, ## required for initialization

                model_type: str = None,  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].
                device = None          
                 ) -> None:

        super().__init__()               
        if (key_cache is not None) or (value_cache is not None) or (past_tok_ids is not None):
            assert isinstance(key_cache, list)
            assert isinstance(value_cache, list)
            assert isinstance(past_tok_ids, list), f"For MMSepCache, if `key_cache` and `value_cache` are given (e.g., provided from legacy `past_key_values`), `past_tok_ids` corresponding to `key_cache` and `value_cache` must also be provided to initialize MMSepCache."

            assert len(key_cache) == len(past_tok_ids), f"The length of `key_cache` ({len(key_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
            assert len(value_cache) == len(past_tok_ids), f"The length of `value_cache` ({len(value_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
        assert layer_num is not None, f"`layer_num` must be provided according to the pretrained model."

        ## For basic parameters & states    
        self.key_cache: List[torch.Tensor] = key_cache if key_cache is not None else []
        self.value_cache: List[torch.Tensor] = value_cache if value_cache is not None else []    

        self.k_seq_dim = k_seq_dim ## The dimension for the seq_len in key states. Typically, 2.
        self.v_seq_dim = v_seq_dim ## The dimension for the seq_len in value states. Typically, 2.

        self.k_slice = self.DIM_TO_SLICE[k_seq_dim]
        self.v_slice = self.DIM_TO_SLICE[v_seq_dim]
        
        self.k_bat_dim_select = self.BAT_DIM_TO_SELECT[k_seq_dim]
        self.v_bat_dim_select = self.BAT_DIM_TO_SELECT[v_seq_dim]
        self._seen_tokens: int = _seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen as well as performing statistics.
        self.layer_num =  layer_num
        self.device = device # Deprecated


        ## For debugging
        self.PRINT_KV_RATIO_INSIDE = PRINT_KV_RATIO_INSIDE
        self.print_KV_inside_per_steps = print_KV_inside_per_steps
        self._print_kv_ratio_count = 0
        self._kept_kv_ratio: List[Tuple[int]] = _kept_kv_ratio if _kept_kv_ratio is not None else []   

        ## For Streaming SepLLM
        self.past_tok_ids: List[torch.Tensor] = past_tok_ids if past_tok_ids is not None else []  ## It saves all the token ids corresponding to the saved KVs for all layers in MMSepCache      
        # self.left_padding_offset = None
        self._set_layer_wise_attribute("init_cache_size", init_cache_size, layer_num)
        self._set_layer_wise_attribute("local_size", local_size, layer_num)
        self._set_layer_wise_attribute("cache_size", cache_size, layer_num)
        self._set_layer_wise_attribute("sep_cache_size", sep_cache_size, layer_num)
        self._set_layer_wise_attribute("sep_exrange", 0, layer_num) # runtime right boundary for separators, excluded
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size), layer_num) # max right boundary for separators, excluded
        self._set_layer_wise_attribute("image_token_length", image_token_length, layer_num)
        self.image_start_pos = image_start_pos if image_start_pos is not None else [35]
        self.mmsep_layer = mmsep_layer if mmsep_layer is not None else layer_num  # layers before mmsep_layer will use mmsep strategy.
        self.SEP_ACCUMULATION = SEP_ACCUMULATION
        self.USE_MAX_SEP_CACHE = USE_MAX_SEP_CACHE
        self.SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH
        

        ### For positional encoding shifting
        self.APPLY_PE_SHIFT = APPLY_PE_SHIFT
        self.APPLY_PES_INSIDE = APPLY_PES_INSIDE

        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None        
        self._shifted_position_ids: List[torch.Tensor] = _shifted_position_ids if _shifted_position_ids is not None else []        
        self._rope_unsqueeze_dim = _rope_unsqueeze_dim
        self._rope_seq_dim = _rope_seq_dim        

        self.pe_dim = pe_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.pe_dim, 2, dtype=torch.int64).float().to(device) / self.pe_dim))
        self.inv_freq = inv_freq
        self.pe_scaling_factor = pe_scaling_factor
        self._sin_cached = None
        self._cos_cached = None

        if model_type is None:
            assert isinstance(separator_token_ids, list), f"`separator_token_ids: List[int]` must be correctly provided for initialization unless `model_type` is properly given, which will auto-fiil `separator_token_ids`."
            assert len(separator_token_ids) > 0, f"`separator_token_ids: List[int]` should NOT be empty."
            for i in range(len(separator_token_ids)):
                assert isinstance(separator_token_ids[i], int), f"The ids in `separator_token_ids` must be of the type `int`."  
            assert isinstance(PADDING_ID, int), f"`PADDING_ID: int` must be correctly provided for initialization unless `model_type` is properly given, which will auto-fiil `PADDING_ID`."
            self.separator_token_ids = separator_token_ids
            self.PADDING_ID = PADDING_ID                               
        else:
            assert isinstance(model_type, str), f"`model_type` should be a `str` or `None`."
            if 'llava' in model_type.lower():
                # print("Debug: For Llava's default separators")
                self.separator_token_ids = [1, 2, 376, 8652, 29889, 29892, 29901, 29915, 29936, 29973, 29991, 31937] # llava v1.5 7b, 13b
                self.PADDING_ID = 0
            elif 'llama' in  model_type.lower():
                # print("Debug: For Llama's default separators")
                self.separator_token_ids = [128000, 13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256, 262] # llama3 8b
                self.PADDING_ID = 128009
            elif ( 'pythia' in model_type.lower() ) or ( 'gpt_neox' in model_type.lower() ):
                # print("Debug: For GPTNeox's default separators")
                self.separator_token_ids = [15, 13, 32, 2, 28, 27, 209, 186, 187, 964, 1157, 3736, 2195, 3706, 1163, 2490,  50276, 586, 4928, 50275]       # pythia 14b
                self.PADDING_ID = 0
            elif 'falcon' in model_type.lower():
                # print(f"Debug: For Falcon's default separators")
                self.separator_token_ids = [25, 23,  42, 12, 38, 37, 193,  4610,  204, 258, 1212, 23787, 466 ]       # falcon-40b
                self.PADDING_ID = 11
            else:
                raise NotImplementedError(f"NOT implemented for the tokenizer of the backbone model type: `{model_type}`. You must provide `separator_token_ids: List[int]` and `PADDING_ID: int` for initialization in this case! ")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_ids: torch.Tensor,
        layer_idx: int,        
        PREFILLING_FLAG: bool = True,
        visual_sep_pos: Optional[List[torch.Tensor]] = None,  ## For Llava v1.5
        query_states: Optional[torch.Tensor] = None,        
        position_ids: Optional[torch.Tensor]=None,                
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        """

        APPLY_PE_SHIFT = self.APPLY_PE_SHIFT
        APPLY_PES_INSIDE = self.APPLY_PES_INSIDE
        SEP_ACCUMULATION = self.SEP_ACCUMULATION
        USE_MAX_SEP_CACHE = self.USE_MAX_SEP_CACHE
        SEP_PADDING_IN_BATCH = self.SEP_PADDING_IN_BATCH
                
        # Update the number of seen tokens
        if layer_idx == 0:
            if input_ids is not None:
                self._seen_tokens += input_ids.shape[-1]
            else:
                self._seen_tokens += key_states.shape[-2]  ## seq_len dimension is -2 for key_states
            
        # [bsz, num_heads, seq_len, head_dim]
        new_kv_pair = (key_states, value_states)
                
        if (key_states.shape[self.k_seq_dim] + self.get_usable_length(layer_idx) < self.cache_size[layer_idx]) or PREFILLING_FLAG:  ## For prefilling

            # Update cache and past token ids                
            self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=False, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            
            if APPLY_PE_SHIFT:                     
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs ) 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)

            ## Count KV usage
            # kv_len_ori = self.get_seq_length(layer_idx)
            # kv_len_cmp = self.get_usable_length(layer_idx)
            # self._update_kv_ratio(kv_len_cmp=kv_len_cmp, kv_len_ori=kv_len_ori, layer_idx=layer_idx)
            
            IS_LAST_TOKEN_SEP = False
            if not PREFILLING_FLAG and len(self.past_tok_ids[layer_idx]) != 0:
                ## This means that we have finished the pre-filling stage at the current layer.
                ## We now look at the last token id in `past_tok_ids[layer_idx]` to see if it is a separator token.
                IS_LAST_TOKEN_SEP = self.past_tok_ids[layer_idx][0, -1].item() in self.separator_token_ids

        else:
            ## Update the KV cache, count KV usage, and compress the KV cache
            offset_init_size_layer = self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=True, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            IS_LAST_TOKEN_SEP = self.past_tok_ids[layer_idx][0, -1].item() in self.separator_token_ids
                        
            if APPLY_PE_SHIFT:                
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs )                 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)
        
        ## v2: use mmsep 
        if IS_LAST_TOKEN_SEP or PREFILLING_FLAG or visual_sep_pos is None or layer_idx < self.mmsep_layer:
            if query_states is not None:
                return self.key_cache[layer_idx], self.value_cache[layer_idx], query_states
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:
            mmsep_key_cache = [
                self.key_cache[layer_idx][:, :, :self.image_start_pos[0], :],
                self.key_cache[layer_idx][:, :, visual_sep_pos, :],
                self.key_cache[layer_idx][:, :, self.image_start_pos[0]+self.image_token_length[layer_idx]:, :]
            ]
            mmsep_key_cache = torch.cat( mmsep_key_cache, dim=2)
            mmsep_value_cache = [
                self.value_cache[layer_idx][:, :, :self.image_start_pos[0], :], 
                self.value_cache[layer_idx][:, :, visual_sep_pos, :],
                self.value_cache[layer_idx][:, :, self.image_start_pos[0]+self.image_token_length[layer_idx]:, :]
            ]
            mmsep_value_cache = torch.cat( mmsep_value_cache, dim=2)
            if query_states is not None:
                return mmsep_key_cache, mmsep_value_cache, query_states
            else:
                return mmsep_key_cache, mmsep_value_cache
            
    
    def update_kv_cache_and_past_tok_ids(self, new_kv_pair: Tuple[torch.Tensor], input_ids: torch.Tensor, layer_idx: int, COMPRESS_KV=False, SEP_ACCUMULATION:bool=True, USE_MAX_SEP_CACHE:bool=False, SEP_PADDING_IN_BATCH:bool=True) -> None:
        """Update the KV cache and past token ids; compress the KV cache if necessary."""

        # 4. 后续decode阶段，每次输入一个token，append到cache后面，并记录 gen_token 的 id 和长度
        if input_ids is not None:
            self.append_past_tok_ids(input_ids, layer_idx)  # 这里 append 是当前解码 step 的 input_ids，只能在这些 token 里丢弃非 sep token 的 kv
        # kv cache 中要丢弃的 token 位置是，past_tok_ids 中非sep 的位置 + init_cache_size

        key, value = new_kv_pair
                
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key)                        
            self.value_cache.append(value)  
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx] , key], dim=self.k_seq_dim)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx] , value], dim=self.v_seq_dim)

        if COMPRESS_KV:
            # no-image-token-ids version
            cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer = self.compress_kv_cache_and_tokids_noimg_layer_wise((self.key_cache[layer_idx], self.value_cache[layer_idx]), layer_idx ,SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH )

            self.set_kv_cache(cmp_past_kv_pairs, layer_idx)
            self.set_past_tok_ids(cmp_past_tok_ids, layer_idx)            
            return offset_init_size_layer
        

    def append_past_tok_ids(self, input_ids: torch.Tensor, layer_idx: int) -> None:
        """Naively append the new `input_ids` to `self.past_tok_ids[layer_idx]`"""    
        
        if len(self.past_tok_ids) <= layer_idx:                        
            self.past_tok_ids.append(input_ids)                     
        else:             
            self.past_tok_ids[layer_idx] = torch.cat([self.past_tok_ids[layer_idx] , input_ids], dim=-1)


    def compress_kv_cache_and_tokids_noimg_layer_wise(self, past_kv_pairs, layer_idx:int ,SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False, SEP_PADDING_IN_BATCH=True ): # my
        key, value = past_kv_pairs
        seq_len = key.size(self.k_seq_dim)
        seq_len_ids = self.past_tok_ids[layer_idx].size(1)  
        offset_init_size_layer = self.init_cache_size[layer_idx]    # original MMSepCache: + left_padding_offset 
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size, bias=0), self.layer_num)
        # self._CHECK_PARAMS_VALIDITY(layer_idx, 0)   # comment out for speed consideration

        if self.sep_exrange[layer_idx] <=0:            
            self.sep_exrange[layer_idx] = offset_init_size_layer

        # initial tokens to keep
        # no initial token ids under llava multimodal setting
        if offset_init_size_layer > 0:
            initial_kv =  self._slice_kv(0, offset_init_size_layer, kv_pair=past_kv_pairs, seq_len=seq_len, _CHECK_IDX=False)

        Before_First_Time_Compress_Flag = (self.sep_exrange[layer_idx] == offset_init_size_layer)  
        ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.

        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:
            past_sep_kv = self._slice_kv(offset_init_size_layer, self.sep_exrange[layer_idx], kv_pair=past_kv_pairs, seq_len=seq_len, _CHECK_IDX=False)
            past_sep_tokids = self._slice_tok_ids(0, self.sep_exrange[layer_idx] - offset_init_size_layer, 
                                                  tok_ids_list=self.past_tok_ids[layer_idx], seq_len=seq_len_ids, _CHECK_IDX=False)
        
        # get the window to be compressed this time
        past_win_kv = self._slice_kv(self.sep_exrange[layer_idx], seq_len-self.local_size[layer_idx], 
                                     kv_pair=past_kv_pairs, seq_len=seq_len, _CHECK_IDX=False)
        past_win_tokids = self._slice_tok_ids(self.sep_exrange[layer_idx]-offset_init_size_layer, seq_len_ids-self.local_size[layer_idx], 
                                              tok_ids_list=self.past_tok_ids[layer_idx], seq_len=seq_len_ids, _CHECK_IDX=False)

        # get the local window to be kept this time
        local_kv = self._slice_kv(seq_len-self.local_size[layer_idx], seq_len, 
                                  kv_pair=past_kv_pairs, seq_len=seq_len, _CHECK_IDX=False)
        local_tokids = self._slice_tok_ids(seq_len_ids-self.local_size[layer_idx], seq_len_ids, 
                                           tok_ids_list=self.past_tok_ids[layer_idx], seq_len=seq_len_ids, _CHECK_IDX=False)

        # get the new sep kv and sep token ids that were just compressed from the past window
        new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num = self.compress_past_win_2_seps(past_win_kv, past_win_tokids, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH) 

        # accumulate past kv and ids
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:  ## Try to accumulate all the seen seps           
            sep_kv, sep_tokids  = self.cat_kv_cache_and_tokids( [ past_sep_kv, new_sep_kv ] ,  [past_sep_tokids, new_sep_tokids ] )                
            new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]  
        else: ## Only keep the newly obtained kv (those just compressed from the past window)
            sep_kv, sep_tokids = new_sep_kv, new_sep_tokids
            sep_len = sep_tokids.shape[-1]

        if USE_MAX_SEP_CACHE: ## Fixed sep cache size, i.e., only keep max_sep_len seps' kv in the cache. 
            if offset_init_size_layer + sep_len > self.max_sep_exidx[layer_idx]:
                max_sep_len = self.max_sep_exidx[layer_idx] - offset_init_size_layer

                sep_kv = self._slice_kv( sep_len-max_sep_len, sep_len, kv_pair=sep_kv, seq_len=sep_tokids.shape[-1], _CHECK_IDX=False)  # True for debug
                sep_tokids = self._slice_tok_ids(  sep_len-max_sep_len, sep_len, tok_ids_list=sep_tokids, seq_len=sep_tokids.shape[-1], _CHECK_IDX=False)
                self.sep_exrange[layer_idx] =  self.max_sep_exidx[layer_idx]  
            else:
                self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len
        
        else:   ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set
            # ! note: this may cause endless kv cache
            self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len
            if self.sep_exrange[layer_idx] > self.max_sep_exidx[layer_idx]:                    
                cache_incremental_gap = self.sep_exrange[layer_idx] - self.max_sep_exidx[layer_idx]
                self.max_sep_exidx[layer_idx] = self.sep_exrange[layer_idx] 
                self.sep_cache_size[layer_idx] = self.sep_cache_size[layer_idx] + cache_incremental_gap
                self.cache_size[layer_idx] = self.cache_size[layer_idx] + cache_incremental_gap

        if offset_init_size_layer > 0:                                
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [ initial_kv, sep_kv, local_kv ] ,  [ sep_tokids, local_tokids ] )
        else:
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [ sep_kv, local_kv ] ,  [ sep_tokids, local_tokids ] )
                
        return cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer


    def compress_kv_cache_and_tokids_layer_wise(self, past_kv_pairs, layer_idx:int ,SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False, SEP_PADDING_IN_BATCH=True ):
        key, value = past_kv_pairs
        seq_len = key.size(self.k_seq_dim)
        assert seq_len == self.get_usable_length(layer_idx), f"The seq_len of cached past key and value states should be the same as the return of `get_usable_length()`, which is {self.get_usable_length(layer_idx)}"

        offset_init_size_layer = self.init_cache_size[layer_idx]    # + left_padding_offset
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size, bias=0), self.layer_num)
        self._CHECK_PARAMS_VALIDITY(layer_idx, 0)   # set left_padding_offset=0 for MMSepCache during inference, because batchsize=1 and no padding

        if self.sep_exrange[layer_idx] <=0:            
            self.sep_exrange[layer_idx] = offset_init_size_layer

        assert seq_len - self.local_size[layer_idx] > self.sep_exrange[layer_idx]
        
        if offset_init_size_layer > 0:                                                       
            initial_kv, initial_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], 0, offset_init_size_layer, seq_len=seq_len, _CHECK_IDX=True )        

        Before_First_Time_Compress_Flag = (self.sep_exrange[layer_idx] == offset_init_size_layer)  ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag: ## To get the old sep kv and sep token ids.           
            past_sep_kv, past_sep_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], offset_init_size_layer, self.sep_exrange[layer_idx], seq_len=seq_len, _CHECK_IDX=True )            
        
        past_win_kv, past_win_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], self.sep_exrange[layer_idx], seq_len - self.local_size[layer_idx], seq_len=seq_len, _CHECK_IDX=True )        
        
        
        local_kv, local_tokids  =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], seq_len - self.local_size[layer_idx], seq_len, seq_len=seq_len, _CHECK_IDX=True )
        
        new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num = self.compress_past_win_2_seps( past_win_kv, past_win_tokids, SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH ) ## To get the new sep kv and sep token ids that were just compressed from the past window
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:  ## Try to accumulate all the seen seps           
            sep_kv, sep_tokids  = self.cat_kv_cache_and_tokids( [ past_sep_kv, new_sep_kv ] ,  [past_sep_tokids, new_sep_tokids ] )                
            new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]  
        else: ## Only keep the newly obtained kv (those just compressed from the past window)
            sep_kv, sep_tokids = new_sep_kv, new_sep_tokids
            sep_len = sep_tokids.shape[-1]            
            assert (SEP_PADDING_IN_BATCH and max_sep_num==sep_len) or ( (not SEP_PADDING_IN_BATCH) and min_sep_num==sep_len)


        if USE_MAX_SEP_CACHE: ## Fixed sep cache size, i.e., only keep max_sep_len seps' kv in the cache. 
            if offset_init_size_layer + sep_len > self.max_sep_exidx[layer_idx]:
                max_sep_len = self.max_sep_exidx[layer_idx] - offset_init_size_layer
                assert sep_kv[0].shape[-2] == sep_tokids.shape[-1], f"The seq_len for seps' KVs and tok_ids should be the same."

                sep_kv, sep_tokids =  self.slice_kv_cache_and_tokids( sep_kv, sep_tokids, sep_len-max_sep_len, sep_len, seq_len = sep_tokids.shape[-1] ,_CHECK_IDX=True )
                self.sep_exrange[layer_idx] =  self.max_sep_exidx[layer_idx]  
            else:
                self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len             

        else:    ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set                           
            self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len
            if self.sep_exrange[layer_idx] > self.max_sep_exidx[layer_idx]:                    
                cache_incremental_gap = self.sep_exrange[layer_idx] - self.max_sep_exidx[layer_idx]
                self.max_sep_exidx[layer_idx] = self.sep_exrange[layer_idx] 
                self.sep_cache_size[layer_idx] = self.sep_cache_size[layer_idx] + cache_incremental_gap
                self.cache_size[layer_idx] = self.cache_size[layer_idx] + cache_incremental_gap

        if offset_init_size_layer > 0:                                
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [initial_kv, sep_kv, local_kv ] ,  [initial_tokids, sep_tokids, local_tokids  ] )
        else:
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [sep_kv, local_kv ] ,  [sep_tokids, local_tokids  ] )
                
        return cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer
            

    def compress_past_win_2_seps(self, past_win_kv: Tuple[torch.Tensor], past_win_tokids: torch.Tensor, MIN_SEP_ALERT: bool=False, SEP_PADDING_IN_BATCH: bool=True ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor, int ]]:
        """Compress the KVs in the past window into the sep cache where only separators' KVs are kept. Padding or Truncating if necessary."""
        sep_index_tensor = torch.zeros_like(past_win_tokids).bool()  # batch x seq_len

        for sp_id in self.separator_token_ids:            
            sep_index_tensor = sep_index_tensor | ( past_win_tokids == sp_id ) # batch x seq_len

        sep_cnt = sep_index_tensor.int().sum(-1)
        min_sep_num = sep_cnt.min()  # the min sep number for the seqs in a batch
        max_sep_num = sep_cnt.max()  # the max sep number for the seqs in a batch

        
        if MIN_SEP_ALERT and not SEP_PADDING_IN_BATCH:
            assert min_sep_num>0, f"The min sep number for each compressing time in a batch should be at least one if `MIN_SEP_ALERT=True` and `SEP_PADDING_IN_BATCH=False`"
                
        batch1_sep_ids_list = []
        batch_size = past_win_tokids.shape[0]
        for b_id in range(batch_size):            
            batch1_sep_ids = past_win_tokids[b_id, sep_index_tensor[b_id]] # #  sep_num
            if SEP_PADDING_IN_BATCH: ## padding
                sep_num = batch1_sep_ids.shape[-1]
                padding_num =  max_sep_num - sep_num                       
                if padding_num > 0:
                    assert padding_num <= past_win_tokids.shape[-1], f"padding_num: {padding_num} should be <= past_win_tokids.shape[-1]:{past_win_tokids.shape[-1]}"
                    batch1_sep_ids = batch1_sep_ids  # #  sep_num
                    batch1_pad_ids = past_win_tokids[b_id, -padding_num:]  # #  padding_num
                    batch1_sep_ids =  torch.cat([batch1_sep_ids, batch1_pad_ids], dim =-1)   ##  max_sep_num                
            else: ## truncating
                batch1_sep_ids = batch1_sep_ids[..., :min_sep_num ]  # #  min_sep_num
            batch1_sep_ids_list.append(batch1_sep_ids)                                                           
            
        new_sep_tokids = torch.stack(batch1_sep_ids_list, dim=0) # #  B x min_sep_num
        key_cache, value_cache = past_win_kv

        batch1_sep_k_list = []
        batch1_sep_v_list = []
        batch1_sep_ids_list = []
        for b_id in range(batch_size):
            batch1_sep_k = self.k_bat_dim_select(key_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_k_list.append(batch1_sep_k)

            batch1_sep_v = self.v_bat_dim_select(value_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_v_list.append( batch1_sep_v )   
        
        sep_k = torch.stack(batch1_sep_k_list, dim=0)  ## batch x head x min_sep_num x dim
        sep_v = torch.stack(batch1_sep_v_list, dim=0)  ## batch x head x min_sep_num x dim                   
        new_sep_kv = (sep_k, sep_v)

        return new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num      


    def apply_shifted_pos_emb(self, layer_idx: int, APPLY_PES_INSIDE: bool, PREFILLING_FLAG: bool, key_states: torch.Tensor, query_states: torch.Tensor, position_ids: torch.Tensor, cache_kwargs: Optional[Dict[str, Any]] = None ) -> torch.Tensor:        
        """Perform positional encoding shifting if required"""
        seq_len = self.get_usable_length(layer_idx)
        keys_to_shift = self.key_cache[layer_idx]
        queries_to_shift = query_states
        assert keys_to_shift.shape[self.k_seq_dim] == seq_len
        
        if cache_kwargs is None:
            cache_kwargs = {}

        if APPLY_PES_INSIDE:           
            if len(self._shifted_position_ids) <= layer_idx:
                self._shifted_position_ids.append(None)

            if PREFILLING_FLAG: ## for prefilling
                assert position_ids.shape[-1] >= seq_len, f"The length of position_ids should be >= the usable length of kv cache when prefilling."                
                self._shifted_position_ids[layer_idx] = position_ids[:, :seq_len].detach()
                shifted_pos_ids = self._shifted_position_ids[layer_idx]

            elif self._shifted_position_ids[layer_idx].shape[-1] >= seq_len:  ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                shifted_pos_ids = self._shifted_position_ids[layer_idx][:, :seq_len].detach()

            elif self._shifted_position_ids[layer_idx].shape[-1] < seq_len:   ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                increased_gap = seq_len - self._shifted_position_ids[layer_idx].shape[-1]
                assert increased_gap < self._shifted_position_ids[layer_idx].shape[-1], f"Normally, for auto-regressive model, the input length for each step should be 1 during generation."

                new_position_ids = self._shifted_position_ids[layer_idx][:, -increased_gap: ] + increased_gap
                self._shifted_position_ids[layer_idx] = torch.cat([self._shifted_position_ids[layer_idx], new_position_ids.detach()], dim=-1)
                shifted_pos_ids = self._shifted_position_ids[layer_idx]
            else:
                raise RuntimeError

            cos, sin = self._get_naive_shifted_cos_sin(
                key_states, shifted_pos_ids, seq_len
            )

            q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
            cos_q, sin_q = cos.index_select(self._rope_seq_dim, q_rope_idx), sin.index_select(self._rope_seq_dim, q_rope_idx)

        else:
            sin = cache_kwargs.get("sin")
            cos = cache_kwargs.get("cos")                         
            sin_q = cache_kwargs.get("sin_q")
            cos_q = cache_kwargs.get("cos_q")    
            shifted_pos_ids = cache_kwargs.get("shifted_pos_ids") 
            assert (sin is not None) and (cos is not None), f"sin and cos matrices should be be provided"
            if sin_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(sin.device)
                sin_q = sin.index_select(self._rope_seq_dim, q_rope_idx)
            if cos_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
                cos_q = cos.index_select(self._rope_seq_dim, q_rope_idx)
            
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        
        # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
        if partial_rotation_size is not None:
            keys_to_shift, keys_pass = (
                keys_to_shift[..., :partial_rotation_size],
                keys_to_shift[..., partial_rotation_size:]
            )
            queries_to_shift, queries_pass = (
                queries_to_shift[..., :partial_rotation_size],
                queries_to_shift[..., partial_rotation_size:]
            )
                                    
        shifted_keys = self._apply_rotary_pos_emb_single(keys_to_shift, cos, sin, shifted_pos_ids, unsqueeze_dim=self._rope_unsqueeze_dim)
        shifted_queries = self._apply_rotary_pos_emb_single(queries_to_shift, cos_q, sin_q, shifted_pos_ids[:,  -queries_to_shift.shape[self.k_seq_dim] : ], unsqueeze_dim=self._rope_unsqueeze_dim)

        if partial_rotation_size is not None:
            shifted_keys = torch.cat( [shifted_keys, keys_pass], dim=-1)
            shifted_queries = torch.cat( [shifted_queries, queries_pass], dim=-1)


        return shifted_keys, shifted_queries


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the seen tokens. A layer index can be optionally passed."""                
        return self._seen_tokens


    def get_usable_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the actual cached states. A layer index must be passed."""         
        if len(self.key_cache) <= layer_idx :
            return 0
        # assert self.key_cache[layer_idx].shape[self.k_seq_dim] == self.value_cache[layer_idx].shape[self.v_seq_dim], f"`self.key_cache` and `self.value_cache` should have the same length."        
        return self.key_cache[layer_idx].shape[self.k_seq_dim]

    def get_initial_pos_offset(self, layer_idx:int = 0) -> int:      
        """Return the number of padding tokens in the record with the most left padding tokens in a batch."""
        assert isinstance(self.PADDING_ID, int), f"`self.PADDING_ID` should be correctly set."
        assert len(self.past_tok_ids) > layer_idx, f"`self.past_tok_ids` for layer {layer_idx} must have been properly set."
                
        past_tok_ids = self.past_tok_ids[layer_idx]
        assert past_tok_ids is not None, f"`past_tok_ids` for layer {layer_idx} should not be None"

        pad_index_tensor = (past_tok_ids == self.PADDING_ID)  ## batch x seq_len
        pad_toks_cnt = pad_index_tensor.int().sum(-1)  ## [batch]
        offset = pad_toks_cnt.max().item()

        return offset

                             
    def get_batch_size(self) -> int:
        """Return the batch size."""
        assert self.key_cache is not None, f"`self.key_cache` should not be None."
        assert self.value_cache is not None, f"`self.value_cache` should not be None."
        assert len(self.key_cache) > 0, f"`self.key_cache` is empty. No batch size is available."
        assert len(self.value_cache) > 0, f"self.value_cache is empty. No batch size is available."

        assert len(self.value_cache) == len(self.key_cache), f"self.value_cache and self.key_cache should be at the same length."
        assert self.value_cache[0].shape[0] == self.key_cache[0].shape[0], f"self.value_cache and self.key_cache should have the same batch size."

        return self.value_cache[0].shape[0]

    def get_kv_pair(self, layer_idx: int = None) -> Tuple[torch.Tensor]:
        assert layer_idx is not None, f"`layer_idx` must be given."

        if (len(self.key_cache) <= layer_idx) and (len(self.value_cache) <= layer_idx ):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
        else:
            raise RuntimeError(f"The KV for layer:{layer_idx} have not been set.")
        return (key, value)


    def set_kv_cache(self, kv_pair: Tuple , layer_idx: int ) -> None:
        self.key_cache[layer_idx] = kv_pair[0]
        self.value_cache[layer_idx] = kv_pair[1]
    
    def set_past_tok_ids(self, tok_ids: torch.Tensor, layer_idx:int) -> None:
        self.past_tok_ids[layer_idx] = tok_ids


    def cat_kv_cache_and_tokids(self, kv_pairs_list: List[Tuple[torch.Tensor]] , tok_ids_list:List[torch.Tensor]) -> Tuple[Union[Tuple[torch.Tensor],torch.Tensor]]:
        
        return self.cat_kv_cache(kv_pairs_list), self.cat_token_ids(tok_ids_list)


    def slice_kv_cache_and_tokids(self, kv_pair:Tuple[torch.Tensor], tok_ids_list:torch.Tensor, start:int, end:int, seq_len:int=None, _CHECK_IDX:bool=True, ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor]]:
                             
        sliced_kv = self._slice_kv(start, end,  kv_pair=kv_pair, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX,)                                    
        sliced_tids = self._slice_tok_ids(start, end, tok_ids_list = tok_ids_list, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX)
        
        return sliced_kv , sliced_tids


    def slice_kv_cache(self, kv_pair:Tuple[torch.Tensor], offset_len:int=0, seq_len:int=None, _CHECK_IDX:bool=True,) -> Tuple[torch.Tensor]:
        """slice kv cache to three parts: 0 to offset_len, offset_len to (seq_len - local), and local_window size """
        assert kv_pair is not None, f"kv_pair must NOT be None when slicing it."
        key_cache = kv_pair[0]
        value_cache = kv_pair[1]

        slice_point_init = offset_len
        slice_point_local = seq_len - self.local_size[0]  # assuming all layers
        assert slice_point_init < slice_point_local , f"slice_point_init: {slice_point_init} must be < slice_point_local: {slice_point_local}"
        
        init_key_cache = self.k_slice(key_cache, 0, slice_point_init)
        init_value_cache = self.v_slice(value_cache, 0, slice_point_init)

        past_win_key_cache = self.k_slice(key_cache, slice_point_init, slice_point_local)
        past_win_value_cache = self.v_slice(value_cache, slice_point_init, slice_point_local)

        local_key_cache = self.k_slice(key_cache, slice_point_local, seq_len)
        local_value_cache = self.v_slice(value_cache, slice_point_local, seq_len)

        return (init_key_cache, init_value_cache), (past_win_key_cache, past_win_value_cache), (local_key_cache, local_value_cache)


    def slice_past_tok_ids(self, tok_ids_list:torch.Tensor, seq_len:int=None, _CHECK_IDX:bool=True,) -> torch.Tensor:
        """slice past_tok_ids to two parts: 0 to (seq_len - local), and local_window size """
        assert tok_ids_list is not None, f"tok_ids_list must NOT be None when slicing it."
        
        
        slice_point_local = seq_len - self.local_size[0]  # assuming all layers

        past_gen_tokids = tok_ids_list[:, 0:slice_point_local]
        local_tokids = tok_ids_list[:, slice_point_local:seq_len]

        return past_gen_tokids, local_tokids
    

    def cat_kv_cache(self, kv_pairs_list: List[Tuple[torch.Tensor]] ) -> Tuple[torch.Tensor]:               
        assert len(kv_pairs_list) >= 1 
        
        if len(kv_pairs_list) == 1 :
            return kv_pairs_list[0]
        else:
            ret = None 
            for i, kv_pair in enumerate(kv_pairs_list): # enumerate all the KVs needed to be cat
                if i == 0:
                    ret = kv_pair
                else:
                    ret = self._cat_kv(ret, kv_pair)
            return ret


    def cat_token_ids(self, tok_ids_list:List[torch.Tensor]  ) -> torch.Tensor :
        assert len(tok_ids_list) >= 1 
        
        return torch.cat(tok_ids_list, dim=-1)     


    def _cat_kv(self, kv_pair_a:Tuple[torch.Tensor],  kv_pair_b:Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:            
        k_a, v_a = kv_pair_a
        k_b, v_b = kv_pair_b
        
        cat_k = torch.cat([k_a, k_b], dim=self.k_seq_dim)
        cat_v = torch.cat([v_a, v_b], dim=self.v_seq_dim)
        return (cat_k, cat_v)


    def _slice_kv(self, start:int, end:int, kv_pair: Tuple[torch.Tensor],   seq_len:int=None, _CHECK_IDX:bool=True)  -> Tuple[torch.Tensor] :
        key_cache = kv_pair[0]
        value_cache = kv_pair[1]

        # if _CHECK_IDX:                                 
        #     assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
        #     start, end = self._CHECK_IDX(start, end, seq_len)
            
        sliced_key_cache = self.k_slice(key_cache, start, end) 
        sliced_value_cache = self.v_slice(value_cache, start, end)

        return ( sliced_key_cache, sliced_value_cache)


    def _slice_tok_ids(self, start:int, end:int, tok_ids_list:torch.Tensor , seq_len:int=None, _CHECK_IDX:bool=False) -> torch.Tensor:
        
        # if _CHECK_IDX:
        #     assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
        #     start, end = self._CHECK_IDX(start, end, seq_len)        
          
        sliced_tok_ids = tok_ids_list[:, start:end]
        return sliced_tok_ids

    def _set_layer_wise_attribute(self, name: str, value: Any, layer_num:int ):
        """Set layer-wise attributes"""
        if isinstance(value, int):        
            setattr(self, name, [value] * layer_num)
        elif isinstance(value, (list, tuple)):
            assert len(value) == layer_num, f"The length of {name}: {len(value)} must be equal to `layer_num`: {layer_num}"
            setattr(self, name, list(value))
        else:
            raise TypeError(f"{name} must be of the type `int` or `list` but got `{type(value)}`")

    def _list_element_add(self, list_a: List, list_b: List, bias: int=0, dtype = int, device = 'cpu') -> List:  
        """Element-wise addition between two lists."""      
        assert len(list_a) == len(list_b), f"The length of `list_a` ({len(list_a)}) must be equal to that of `list_b` ({len(list_b)})."
        tensor_c = torch.tensor(list_a, dtype=dtype, device=device) + torch.tensor(list_b, dtype=dtype, device=device) + torch.tensor([bias], dtype=dtype, device=device)
        return tensor_c.int().tolist()
        
    def _CHECK_IDX(self, start: int = 0, end: int = 100, seq_len: int = 1000):
        assert isinstance(start, int) and isinstance(end, int) and isinstance(seq_len, int), f"`start`, `end`, `seq_len` must be `int`."
        assert seq_len>0 , f"`seq_len` must > 0"
        
        if start <0 :
            start = start % seq_len
        if end < 0 :
            end = end % seq_len
        assert (start >=0) and (start < seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert (end >= 0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}"

        return start,end

    def _CHECK_PARAMS_VALIDITY(self, layer_idx:int, left_padding_offset:int):
        assert len(self.cache_size) > layer_idx
        assert len(self.init_cache_size) > layer_idx
        assert len(self.sep_cache_size) > layer_idx
        assert len(self.max_sep_exidx) > layer_idx
        assert len(self.local_size) > layer_idx

        assert self.cache_size[layer_idx] > 0 , f"`self.cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] >= 0 , f"`self.init_cache_size` for layer:{layer_idx} must be greater than (equal to) 0"
        assert self.local_size[layer_idx] > 0 , f"`self.local_size` for layer:{layer_idx} must be greater than 0"
                    
        assert self.sep_cache_size[layer_idx] > 0 , f"`self.sep_cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.max_sep_exidx[layer_idx] > 0 , f"`self.max_sep_exidx` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] + self.sep_cache_size[layer_idx] + self.local_size[layer_idx] + left_padding_offset < self.cache_size[layer_idx], f"`init_cache_size` ({self.init_cache_size[layer_idx]}) + `sep_cache_size` ({self.sep_cache_size[layer_idx]}) + `local_size` ({self.local_size[layer_idx]}) + `left_padding_offset` ({left_padding_offset}) for layer {layer_idx} should be less than `cache_size`:({self.cache_size[layer_idx]}) for layer {layer_idx}, i.e., a + s + w + (left_padding_offset) < c. Please increase `cache_size` if applicable."
        


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_single(self, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """        
        cos = cos.unsqueeze(unsqueeze_dim)   # batch x seq_len x dim  --> batch x 1 x seq_len x dim
        sin = sin.unsqueeze(unsqueeze_dim)        
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return  k_embed


    def _get_naive_shifted_cos_sin(self, x: torch.Tensor, position_ids: torch.Tensor=None, seq_len=None):
        # x: [batch, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # backwards compatibility
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin
    

    def _get_scaled_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _get_dynamicNTK_scaling_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO: this may break with compilation

        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _update_kv_ratio(self, kv_len_cmp:int, kv_len_ori:int, layer_idx: int=0) -> None:
        """Update the KV ratios which are for statistics and debugging."""
        if len(self._kept_kv_ratio) <= layer_idx:
            self._kept_kv_ratio.append( (kv_len_cmp,  kv_len_ori ) )    
        else:
            old_kv_len_cmp = self._kept_kv_ratio[layer_idx][0]
            old_kv_len_ori = self._kept_kv_ratio[layer_idx][1]
            self._kept_kv_ratio[layer_idx] = (old_kv_len_cmp + kv_len_cmp,  old_kv_len_ori + kv_len_ori )

    @classmethod ## Deprecated
    def from_legacy_cache(cls, 
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,                          
                init_cache_size: Union[int, List] = 4,        
                sep_cache_size: Union[int, List] = 64,
                local_size: Union[int, List]=256, 
                cache_size: Union[int, List]=512,
                image_token_length: Union[int, List]=576,
                image_start_pos: List[int] = None,
                mmsep_layer: int = None,
                SEP_ACCUMULATION: bool = True,
                USE_MAX_SEP_CACHE: bool = False,
                SEP_PADDING_IN_BATCH: bool = False,
                separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided. set it to `[-1]` to degrade MMSepCache to StreamingLLM's SinkCache
                PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

                ## For inheritance & initialization states
                past_tok_ids: List[torch.Tensor] = None,  ## It saves all the token ids corresponding to the saved KVs for all layers in MMSepCache.                
                key_cache: List[torch.Tensor] = None,          
                value_cache: List[torch.Tensor] = None,

                ## For debugging
                PRINT_KV_RATIO_INSIDE: bool = False,
                print_KV_inside_per_steps: int = 1000,   
                _seen_tokens: int = 0, 
                _kept_kv_ratio: List[Tuple[int]] = None,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT: bool = False,
                APPLY_PES_INSIDE: bool = True,
                _shifted_position_ids:  List[torch.Tensor] = None,
                _rope_unsqueeze_dim: int = 1, ## The unsqueeze_dim when applying RoPE.
                _rope_seq_dim: int=1, ## The seq_len dimension for the `cos` or `sin` tensors.
                pe_scaling_factor:float = 1.0,
                pe_dim:int=128, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                max_position_embeddings: int = 8192, 
                base: int=10000,  ## The base for RoPE.               
                
                ## For basic transformer architecture
                k_seq_dim: int=2, ## The dimension for seq_len in key tensors
                v_seq_dim: int=2, ## The dimension for seq_len in value tensors
                layer_num: int = None, ## required for initialization

                model_type: str = None,  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].
                device = None    
    ) -> "MMSepCache":

        if past_key_values is not None:
            assert len(past_key_values)==0, f"`from_legacy_cache` function is deprecated. You can only use it when `past_key_values=None` or `past_key_values` is empty, in which case, `from_legacy_cache` is equivalent to the `__init__` function."        
            past_key_values = None

        assert past_key_values is None, f"`from_legacy_cache` function is deprecated. You can only use it when `past_key_values=None` or `past_key_values` is empty, in which case, `from_legacy_cache` is equivalent to the `__init__` function."        
        
        if past_key_values is not None: ## Deprecated
            key_cache = []
            value_cache = []               
            
            for i, kv in enumerate(past_key_values):
                if i == 0:
                    past_tok_ids = [] if len(kv) == 4  else past_tok_ids       

                if len(kv) == 4:
                    k, v, p_tok_ids, _seen_tokens  = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    past_tok_ids.append(p_tok_ids)
                    _seen_tokens = _seen_tokens
                elif len(kv) == 2:
                    k, v = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    
        cache = cls(
                init_cache_size=init_cache_size,        
                sep_cache_size=sep_cache_size,
                local_size=local_size, 
                cache_size=cache_size,
                image_token_length=image_token_length,
                image_start_pos=image_start_pos,
                mmsep_layer=mmsep_layer,
                SEP_ACCUMULATION=SEP_ACCUMULATION,
                USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE,
                SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH,
                separator_token_ids=separator_token_ids,
                PADDING_ID=PADDING_ID,

                ## For inheritance & initialization states
                past_tok_ids=past_tok_ids,  ## It saves all the token ids corresponding to the saved KVs for all layers in MMSepCache        
                key_cache=key_cache,          
                value_cache=value_cache,

                ## For debugging
                PRINT_KV_RATIO_INSIDE=PRINT_KV_RATIO_INSIDE,
                print_KV_inside_per_steps=print_KV_inside_per_steps,   
                _seen_tokens=_seen_tokens, 
                _kept_kv_ratio=_kept_kv_ratio,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT=APPLY_PE_SHIFT,
                APPLY_PES_INSIDE=APPLY_PES_INSIDE,
                _shifted_position_ids=_shifted_position_ids,
                _rope_unsqueeze_dim=_rope_unsqueeze_dim,
                _rope_seq_dim=_rope_seq_dim, 
                pe_scaling_factor=pe_scaling_factor,
                pe_dim=pe_dim,
                max_position_embeddings=max_position_embeddings, 
                base=base,                 
                
                ## For basic transformer architecture
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                layer_num=layer_num,
                
                model_type=model_type,  
                device=device,   
        )

        return cache

    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]: ## Deprecated
        """Deprecated: Converts the `SepCache` instance into the legacy cache format, i.e., tuple."""
        print(">>>>>>>>>>>Warnings: Please try to avoid using this deprecated `to_legacy_cache` function since it will drop many useful parameters or states in MMSepCache.<<<<<<<<<<<")
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.past_tok_ids[layer_idx], self._seen_tokens), )
        return legacy_cache


    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        if self.key_cache is not None:
            return len(self.key_cache)
        else:
            return 0

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None