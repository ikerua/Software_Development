# Machine Learning-Oriented Software Development Project

**Authors**: Iker Urdiroz & Joaquín Orradre

## Introduction
This repository contains an educational project developed as part of the course **Software Development Oriented to Machine Learning**. The goal is to apply good software engineering practices to a machine learning workflow, from data acquisition to experimentation and reproducibility. Specifically, we are going to implement and analyse a regression task using a house pricing dataset.

You can visit code documentation on the following [page](https://ikerua.github.io/Software_Development/)

## Dataset
Our House Pricing Dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/prokshitha/home-value-insights). License is not directly specified in the website, but you can consult the [DOI](https://doi.org/10.34740/kaggle/dsv/9334344).

**Features:**
- *Square_Footage:* The size of the house in square feet. Larger homes typically have higher prices.
- *Num_Bedrooms:* The number of bedrooms in the house. More bedrooms generally increase the value of a home.
- *Num_Bathrooms:* The number of bathrooms in the house. Houses with more bathrooms are typically priced higher.
- *Year_Built:* The year the house was built. Older houses may be priced lower due to wear and tear.
- *Lot_Size:* The size of the lot the house is built on, measured in acres. Larger lots tend to add value to a property.
- *Garage_Size:* The number of cars that can fit in the garage. Houses with larger garages are usually more expensive.
- *Neighborhood_Quality:* A rating of the neighborhood’s quality on a scale of 1-10, where 10 indicates a high-quality neighborhood. Better neighborhoods usually command higher prices.
- *House_Price (Target Variable):* The price of the house, which is the dependent variable you aim to predict.

## Setup Instructions

**Dataset**

The dataset is already present in our repository in the following path:
  ```bash
  data/raw/
  ```
**Installation**

Binary installers for the latest released version are available at the [Test Python Package Index (Test PyPI)](https://test.pypi.org/project/house-price-prediction-joaquin-iker/):
  ```bash
  pip -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ house-price-prediction-joaquin-iker
  ```         
