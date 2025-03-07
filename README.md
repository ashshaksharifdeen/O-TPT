# O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models (CVPR 2025) 

Our contributions are summarized as follows:
  - We provide new insights underlying the suboptimal performance of an existing top-performing calibration method for test-time prompt tuning
  - We propose a novel approach (named `O-TPT`) for calibrating test-time prompt tuning for VLMs by enforcing orthogonality constraints. This is accomplished by introducing orthogonal regularization on the textual features.

  - We perform an extensive evaluation to validate our approach on various datasets and across different baselines. Results reveal that `O-TPT` provides consistent gains over the state-of-the-art methods in overall average calibration performance with several different baselines. Moreover, our `O-TPT` provides better calibration performance than the zero-shot CLIP which reveals improved calibration compared to existing SOTA (include-picture below).

[[Paper]()] [[arXiv]()]

## Contents

1. [Installation](#installation) 
2. [Datasets](#datasets)
3. [Run Experiments](#run-experiments)
4. [Main Results](#main-results)
5. [Acknowledgement](#acknowledgement)
6. [Citation](#citation)
7. [Contact](#contact)

## Installation

## Datasets
We have conducted main experiments on fine-grained and natural distribution shift datasets:

- **Fine-grained datasets**:  
  1. ImageNet  
  2. Flower102  
  3. OxfordPets  
  4. SUN397  
  5. DTD  
  6. Food101  
  7. StanfordCars  
  8. Aircraft  
  9. UCF101  
  10. EuroSAT  
  11. Caltech101  

- **Natural distribution shift datasets**:  
  1. ImageNet-V2  
  2. ImageNet-A  
  3. ImageNet-R  
  4. ImageNet-Sketch

Follow this repository for datasets preparation: [TPT](https://github.com/azshue/TPT) 

## Run Experiments

## Main Results

## Acknowledgement

## Citation

## Contact
