Machine Learning-Oriented Software Development Project

## Introduction

This repository contains an educational project developed as part of the course **Software Development Oriented to Machine Learning**. The goal is to apply good software engineering practices to a machine learning workflow, from data acquisition to experimentation and reproducibility.

You can visit code documentation on the following [page](https://ikerua.github.io/Software_Development/)

## Setup Instructions

1. **Download the dataset**

   - Obtain the dataset required for the project.[Kaggle Dataset](https://www.kaggle.com/datasets/prokshitha/home-value-insights)
   - Place it inside the following directory:
     ```bash
     data/raw/
     ```
2. **Install dependencies**

   - From the project's root directory, open a terminal and run:
     ```bash
     uv sync
     ```
3. **Run the Jupyter Notebook**

   - Once the environment is set up, navigate to the `notebooks/` folder.
   - Open and execute the notebook to reproduce the experiments and results.

## Notes

- Ensure you have `uv` installed on your system before running the synchronization command.
- The dataset is not included in this repository due to storage constraints.
