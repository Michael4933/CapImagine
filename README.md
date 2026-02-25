# Imagination Helps Visual Reasoning, But Not Yet in Latent Space
[You Li](https://scholar.google.com.hk/citations?user=RZ5bOS0AAAAJ&hl=zh-CN), [Chen Chi*](https://openreview.net/profile?id=~Chi_Chen1), [Yanghao Li](https://openreview.net/profile?id=~Yanghao_Li2), [Fanhu Zeng](https://aurorazengfh.github.io/) (Looking for Ph.D.), [Kaiyu Huang](https://openreview.net/profile?id=~Kaiyu_Huang1), Jinan Xu, Maosong Sun


-----

<a href='https://arxiv.org/abs/2501.05767'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
<a href='https://huggingface.co/Michael4933/CapImagine-7B'><img src='https://img.shields.io/badge/Model-Huggingface-red'></a> 
<a href='https://huggingface.co/datasets/Michael4933/CapImagine-Data'><img src='https://img.shields.io/badge/Benchmark-Huggingface-yellow'></a>

This repository hosts the implementation and analysis details of CapImagine. It starts from systematic investigation on the internal mechanisms of latent-space visual reasoning methods through causal mediation analysis followed by a text-space imagination method exhibiting better causal effect and higher performance.  
We believe our study provides a rigorous investigation of current latent visual reasoning methods and offers guidance toward developing more faithful, interpretable, and effective latent reasoning approaches.

-----------

## News
* **[2026.02.26]** 🌞🌞🌞 Our paper has been released at [Arxiv](https://arxiv.org/abs/2501.05767) and [Huggingface Daily Papers](https://arxiv.org/abs/2501.05767)!
* **[2025.02.26]**  🌟🌟🌟 The model weights and datasets are now available on HuggingFace! 🤗 Download and have a try at [Huggingface Model](https://huggingface.co/Michael4933/CapImagine-7B)&[Huggingface Dataset](https://huggingface.co/Michael4933/CapImagine-Data)!

## 🔍Abstract
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

## Causal Mediator Analysis
<img width="8825" height="5024" alt="图片2" src="https://github.com/user-attachments/assets/9f539d92-99d8-454b-ac54-d34ac9053fe7" />
<details><summary> <b> Inter-Instance Analysis </b> </summary>
<p>
lalala
</p>

</details>

## Text-Space Imagination VS Latent-Based Imagination
<img width="7245" height="3320" alt="图片3" src="https://github.com/user-attachments/assets/725c9c04-acbf-4687-9503-04fc361454d0" />

## Technical Illustration of the Manipulation on Latent Tokens

## Latent Visual Reasoning Related Papers
