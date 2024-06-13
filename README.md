# Handwritten Mathematical Expression Recognition

This project focuses on improving the Syntax-Aware Network (SAN) for handwritten mathematical expression recognition. The improvements include modifications to the neural network structure and enhancements to the tree reconstruction algorithm, resulting in higher accuracy and better performance.

## Introduction

Handwritten mathematical expression recognition (HMER) is a challenging task with many potential applications. The Syntax-Aware Network (SAN) incorporates syntax information into the recognition process, enhancing the structural prediction of HMEs.

## Installation

To run this project, you need to have Python and the necessary dependencies installed. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use this project, follow these steps:

1. Clone the repository:

2. Navigate to the project directory:
   ```bash
   
   ```

3. Run the training script:
   ```bash
   bash train.sh
   ```

4. Evaluate the model:
   ```bash
   bash inference.sh
   ```

## Ablation Study

This project includes a series of ablation experiments to evaluate the contributions of different modifications independently. The experiments compare the following configurations:

- Baseline Model
- Network Modification Only
- Algorithm Improvement Only
- Combined Modifications

## Results

The results of the ablation study are summarized below:

| Model Configuration     | CROHME 2014 Accuracy (\%) | CROHME 2016 Accuracy (\%) | CROHME 2019 Accuracy (\%) |
|-------------------------|---------------------------|---------------------------|---------------------------|
| Baseline Model          | 56.2                      | 53.6                      | 53.5                      |
| Network Modification Only | 57.2                    | 53.8                      | 53.6                      |
| Algorithm Improvement Only | 57.4                   | 53.6                      | 53.5                      |
| Combined Modifications  | 57.8                      | 53.8                      | 53.6                      |

## Contributors

- **Yejie Guo:** Responsible for the design and improvement of the model and algorithm, conducting experiments, and writing the experimental report.
- **Jiaguo Tian:** Responsible for part of the testing and writing of the report.

## References

- Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai, "Syntax-Aware Network for Handwritten Mathematical Expression Recognition," *arXiv preprint arXiv:2203.01601*, 2022. [arXiv:2203.01601](https://arxiv.org/abs/2203.01601)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
