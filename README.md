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
| Method                          | Metric  | INet  | DfID  | FLW   | Food  | SUN   | Air   | Pets  | Calt  | UCF   | SAT   | Car   | Avg   |
|---------------------------------|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Zero Shot                       | Acc.    | 66.7  | 44.3  | 67.3  | 83.6  | 62.5  | 23.9  | 88.0  | 92.9  | 65.0  | 41.3  | 63.7  | 63.7  |
|                                 | ECE     | 2.12  | 8.50  | 3.00  | 2.39  | 2.58  | 4.37  | 5.0   | 3.59  | 3.89  | 4.29  | 13.89 | 4.23  |
| TPT                             | Acc.    | 69.0  | 46.7  | 69.0  | 84.7  | 63.4  | 23.4  | 87.1  | 93.2  | 64.8  | 42.1  | 65.3  | 64.8  |
|                                 | ECE     | 10.6  | 21.2  | 13.5  | 3.98  | 11.8  | 5.77  | 4.51  | 3.54  | 12.4  | 3.82  | 5.16  | 11.6  |
| C-TPT                           | Acc.    | 68.5  | 46.0  | 69.8  | 83.7  | 63.4  | 24.85 | 88.0  | 93.3  | 65.7  | 42.3  | 64.57 | 64.57 |
|                                 | ECE     | 3.15  | 11.9  | 5.04  | 3.8   | 4.24  | 4.85  | 3.23  | 2.94  | 4.12  | 2.4   | 1.59  | 5.13  |
| Robust-adapt-SaLs-CTPT          | Acc.    | 68.04 | 45.51 | 69.43 | 83.18 | 64.38 | 23.94 | 88.12 | 93.3  | 65.32 | 43.05 | 65.48 | 64.55 |
|                                 | ECE     | 2.63  | 14.56 | 2.74  | 1.26  | 3.56  | 3.21  | 3.16  | 3.78  | 3.36  | 3.28  | 2.52  | 5.69  |
| Robust-adapt-Penalty-CTPT       | Acc.    | 68.04 | 45.69 | 69.55 | 83.28 | 63.27 | 23.91 | 87.93 | 93.7  | 65.04 | 44.06 | 65.44 | 64.48 |
|                                 | ECE     | 2.13  | 12.39 | 5.27  | 3.35  | 4.87  | 4.31  | 4.56  | 2.29  | 2.72  | 2.15  | 1.67  | 5.16  |
| Robust-adapt-ZS-CTPT            | Acc.    | 68.01 | 45.63 | 69.55 | 83.25 | 64.43 | 23.88 | 88.03 | 93.31 | 65.24 | 44.26 | 65.45 | 64.51 |
|                                 | ECE     | 3.01  | 12.35 | 4.94  | 3.8   | 5.16  | 4.31  | 2.06  | 3.4   | 2.17  | 1.23  | 1.7   | 5.03  |
| O-TPT (Ours)                    | Acc.    | 67.33 | 45.68 | 70.07 | 84.13 | 64.23 | 23.64 | 87.95 | 93.95 | 65.24 | 43.24 | 64.41 | **64.41** |
|                                 | ECE     | 1.96  | 7.88  | 3.87  | 1.46  | 4.93  | 3.68  | 1.9   | 3.38  | 12.98 | 1.78  | **4.21** |


## Acknowledgement
We are thankful to the authors of [TPT](https://github.com/azshue/TPT), [C-TPT](https://github.com/hee-suk-yoon/C-TPT?tab=readme-ov-file), and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for their open-source contributions.

## Citation
If you find our work useful for your research, please consider citing it:

## Contact
If you need any further clarification, please feel free to contact me at [ashshaks@gmail.com](mailto:ashshaks@gmail.com).

