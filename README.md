<h1 align="center">Imagination Helps Visual Reasoning, But Not Yet in Latent Space</h1>

<p align="center">
  <a href="https://scholar.google.com.hk/citations?user=RZ5bOS0AAAAJ&hl=zh-CN">You Li</a>,
  <a href="https://openreview.net/profile?id=~Chi_Chen1">Chen Chi*</a>,
  <a href="https://openreview.net/profile?id=~Yanghao_Li2">Yanghao Li</a>,
  <a href="https://aurorazengfh.github.io/">Fanhu Zeng</a> (📢Looking for PhD!),
  <a href="https://openreview.net/profile?id=~Kaiyu_Huang1">Kaiyu Huang</a>,
  Jinan Xu,
  Maosong Sun
</p>


<p align="center">
<a href='https://arxiv.org/abs/2501.05767'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
<a href='https://huggingface.co/Michael4933/CapImagine-7B'><img src='https://img.shields.io/badge/Model-Huggingface-red'></a> 
<a href='https://huggingface.co/datasets/Michael4933/CapImagine-Data'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'></a>

-----------
This repository hosts the implementation and analysis details of CapImagine. It starts from systematic investigation on the internal mechanisms of latent-space visual reasoning methods through causal mediation analysis followed by a text-space imagination method exhibiting better causal effect and higher performance.  
We believe our study provides a rigorous investigation of current latent visual reasoning methods and offers guidance toward developing more faithful, interpretable, and effective latent reasoning approaches.
</p>

## 📰News
* **[2026.02.26]** 🌞🌞🌞 Our paper has been released at [Arxiv](https://arxiv.org/abs/2501.05767) and [Huggingface Daily Papers](https://arxiv.org/abs/2501.05767)!
* **[2025.02.26]**  🌟🌟🌟 The model weights and datasets are now available on HuggingFace! 🤗 Download and have a try at [Huggingface Model](https://huggingface.co/Michael4933/CapImagine-7B) & [Huggingface Dataset](https://huggingface.co/Michael4933/CapImagine-Data)!

## 📝Abstract
Latent visual reasoning aims to mimic human's *imagination* process by meditating through hidden states of Multimodal Large Language Models.
  While recognized as a promising paradigm for visual reasoning, the underlying mechanisms driving its effectiveness remain unclear.
  Motivated to demystify the true source of its efficacy, we investigate the validity of latent reasoning using Causal Mediation Analysis. 
  We model the process as a causal chain: the input as the treatment, the latent tokens as the mediator, and the final answer as the outcome. 
  Our findings uncover two critical disconnections: 
  (a) **Input-Latent Disconnect**: dramatic perturbations on the input result in negligible changes to the latent tokens, suggesting that latent tokens do not effectively attend to the input sequence.
  (b) **Latent-Answer Disconnect**: perturbations on the latent tokens yield minimal impact on the final answer, indicating the limited causal effect latent tokens imposing on the outcome.
  Furthermore, extensive probing analysis reveals that latent tokens encode limited visual information and exhibit high similarity.
  Consequently, we challenge the necessity of latent reasoning and propose a straightforward alternative named *CapImagine*, which teaches the model to explicitly *imagine* using text.
  Experiments on vision-centric benchmarks show that *CapImagine* significantly outperforms complex latent-space baselines, highlighting the superior potential of visual reasoning through explicit imagination.

## 🔍Causal Mediator Analysis
Given input $X$, intermediate latent steps $Z$ and the final answer $Y$, we frame the whole reasoning process as $X \rightarrow Z \rightarrow Y$. Consequently, we conduct causal mediator analysis on the role of latent tokens by seperately intervening on input $X$ and latent steps $Z$. We investigate the behaviour of these latent tokens through similarity calculation, probing analysis and etc. Full empirical analysis could be found in section3 of our original [paper](https://arxiv.org/abs/2501.05767).  

<img width="8825" height="5024" alt="图片2" src="https://github.com/user-attachments/assets/9f539d92-99d8-454b-ac54-d34ac9053fe7" />  

<br>  
<details><summary> <b> More Inter-Instance Analysis Results </b> </summary>
<p>
  We provide the behaviour pattern over the entire latent sequence for better visualization and deeper analysis.  
  
  **Monet** *(Latent_Size=10)*    
  Measured by cosine similarity, the latent tokens under different inputs carry distinctive semantics at first, but rapidly collapses into a cluster of homogeneous representations, failing to progressively and continually conduct genuine reasoning.
  <details><summary> <b> Visualization </b> </summary>
<p>
  
#### Index progressively increases from left to right, top to bottom.
  <img width="2983" height="1395" alt="image" src="https://github.com/user-attachments/assets/66470353-7575-4850-9578-b981fc624b2d" />
</p>
  </details>
  
  **LVR** *(Latent_Size=8)*   
  During the process of latent reasoning, latent tokens from LVR preserves some extent of distinctiveness by comparison. For several input sequences, the latent tokens exhibits low cosine similarity with others even at the last latent step. However, the majority of latent tokens still exhibits high degreee of degeneration.
  <details><summary> <b> Visualization </b> </summary>
<p>
  
#### Index progressively increases from left to right, top to bottom.
  <img width="2973" height="1628" alt="image" src="https://github.com/user-attachments/assets/45ab2a03-f411-4d76-a56d-3934341c4cb6" />
</p>
  </details>
  
  **Mirage** *(Latent_Size=4)*  
  Due to the highly compressed supervision signal in Mirage, latent tokens at different input sequence and at different indexes are all highly similar, failing to carry rich semantics.
  <details><summary> <b> Visualization </b> </summary>
<p>
  
#### Index progressively increases from left to right, top to bottom.
  <img width="2979" height="763" alt="image" src="https://github.com/user-attachments/assets/7481bd1b-d886-4202-bad7-6bda2ecac382" />
</p>
  </details>


</p>

</details>

From our systematic causal mediator analysis, we could obtain 3 key findings,

> [!IMPORTANT]
> **Findings-1🚀**: **Latent tokens are similar across instances and tasks, and progressively collapse into highly identical states**.
> 
> **Findings-2🚀**: **Fundamental change on latent tokens $Z$ only results in minimal change on answers $Y$**.
> 
> **Findings-3🚀**: **Latent tokens encode limited visual semantics and are insufficient for accurate answer derivation.**

In summary, these **highly homogeneous latent tokens contribute marginally** to the final prediction. The model potentially adopts an **implicit shortcut** circumventing the latent visual reasoning pathway. So far, the full potential of latent tokens in current methods has not yet been fully discovered, and the latent tokens are behaving similarly with **soft prompt or placeholders** instead of active carrier of visual imagination or reasoning.


## 🔥Text-Space Imagination VS Latent-Based Imagination
The essence of visual imagination primarily lies in *interleaved multimodal reasoning*, where internal visual thought could be explicitly outlined and evolve alongside the textual reasoning chain. Existing Latent Visual Reasoning methods attempt to internalize such visual thoughts into latent tokens. However, these latent representations fail to preserve meaningful visual semantics and contribute little to downstream reasoning.   

<img width="7245" height="3320" alt="图片3" src="https://github.com/user-attachments/assets/725c9c04-acbf-4687-9503-04fc361454d0" />

Motivated by this limitation, we explore whether text-space reasoning can more effectively retain the essential information embedded in interleaved data and support visual imagination. Instead of relying on latent variables, we convert the semantic changes introduced by intermediate images into textual captions. This forces the model to imagine visual transformations over the original image through an explicit text-space reasoning chain. Our dataset is currently available at 🤗[CapImagine-Data](https://huggingface.co/Michael4933/CapImagine-Data), which consists of a json file and zip file with all images. The finetuned model based on Qwen2.5-VL-7B has been released at 🤗[CapImagine-7B](https://huggingface.co/Michael4933/CapImagine-7B), whose inference could be implemented through the official code from 🤗[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

<img width="2098" height="930" alt="image" src="https://github.com/user-attachments/assets/984d565d-f0aa-4822-a293-0935f0cbae26" />


### Cases

## 🔧Code Implementation
The main logits of the manipulation of latent tokens happen as illustrated below. We modify the method of `generate` in `transformers.generation.utils`, and further alter the function of `self._sample` to adapt it to supporting latent-based reasoning.

<div style="max-height: 300px; overflow-y: auto;">  
  
```python
class GenerationMixin:
    ...
    def _sample(
        ...
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        ...
        
        # Latent Mode Core Section
        in_latent_mode = False
        is_prefill = True
        latent_start = False
        latent_end = False
        latent_num = 0
        MAX_LATENT_LEN = 10 # latent size
        latent_start_idx = 151666 # token index in vocabulary
        latent_end_idx = 151667
        latent_pad_idx = 151665
        all_hidden_states = []

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # adapatively change between inputting input_ids or inputs_embeds for latent reasoning
            if not in_latent_mode:              
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            else:
                if latent_start:
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    latent_start = False
                else:
                    model_inputs = self.prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, **model_kwargs)
                    assert model_inputs.get("inputs_embeds") is not None
                    model_inputs['input_ids'] = None
            
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True, generate_mode=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True, generate_mode=True, latent_hidden_states=None)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                ...

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
            next_token_strs = [token.item() for token in next_tokens]

            # start from earliest and end after last in batch
            for i, token_str in enumerate(next_token_strs):
                if token_str == latent_start_idx:
                    in_latent_mode = True
                    latent_start = True
                    latent_end = False
                
                elif token_str == latent_end_idx:
                    latent_num = 0
                    latent_end = True
                    in_latent_mode = False
                
                elif in_latent_mode:
                    next_tokens[i] = latent_pad_idx
                    
            if in_latent_mode and not latent_end and not latent_start:
                latent_num += 1

                if latent_num > MAX_LATENT_LEN:
                    latent_end = True
                    in_latent_mode = False
                    latent_start = False
                    next_tokens[:] = latent_end_idx
                    latent_num = 0
                
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)   
            
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # in latent mode, use latent embeds
            if in_latent_mode and not latent_start:
                inputs_embeds = outputs.hidden_states # essential, capturing the output hidden states from last step and sending inputting it to the next step
                all_hidden_states.append(inputs_embeds.squeeze(0)) # accumulate all latent embedding
            
            del outputs

        if len(all_hidden_states) != 0:
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
        
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids # input_ids, all_hidden_states if you want to return latent embeddings
```
</div>

## Latent Visual Reasoning Related Papers
