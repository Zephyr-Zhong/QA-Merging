<!-- preview markdown: ctrl+shift+v -->
# RPAM

**Instruction**

This repository contains the code of the paper **QA-Merging: Query-Adaptive Reasoning via Layer Selective Model Merging** submitted to ACL.

- The code for our merging method can be found in `src`.
- The code for evaluation can be found in `evaluation/script`.

We employ [MergeKit](https://github.com/arcee-ai/mergekit) to merge the Long-CoT and Short-CoT models for initialization and training-free merging baselines. 

**Acknowledgements**: this repository uses codes and resources from [Prodistill](https://github.com/JingXuTHU/Scalable_Model_Merging_with_Progressive_Layerwise_Distillation), [AReal](https://github.com/inclusionAI/AReaL).
