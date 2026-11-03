# Diabetes Prediction with a NumPy MLP

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

A neural network written from scratch in NumPy to predict diabetes on the Pima Indians dataset.

## What is implemented

- Configurable `NeuralNetwork` class with an `[8, 16, 8, 1]` architecture
- Backpropagation with SGD and Adam optimizers, plus L2 regularization
- Median imputation of invalid zero values, standardization, and train/validation/test splits

## Results

| Optimizer | Test accuracy | Macro F1-score |
| --- | --- | --- |
| SGD | 0.73 | 0.71 |
| Adam | 0.73 | 0.71 |

Results come from a test set of 154 patients. The positive class is harder to predict, with an F1-score of 0.62.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook pima_diabetes_mlp.ipynb
```

## Author

**Yassine Houari** · [LinkedIn](https://www.linkedin.com/in/yassine-houari-/)
