# Active Feature Acquisition (AFA) on Data Streams Workshop @ IE 2025

In this workshop you will learn how to run AFA experiments on data streams using the riverml library.

🗂️ Workshop Overview
1. Background on Active Feature Acquisition (30min)
2. Introduction to the Framework (10min)
3. Coding Exercises 1 (20min)
4. Coffee Break (30min)
5. Coding Exercises 2 (20-30min)
6. Presentation of Solutions and Discussion (20min)



## 💻 Getting Started
### 1. Prerequisites

Before you begin, ensure you have the following installed:

Pycharm IDE: [Download here](https://www.jetbrains.com/pycharm/download/?section=windows)

Python 3.13: [Download here](https://www.python.org/downloads/release/python-3135/)

(Code of Framework is maintained [here](https://github.com/cbeyer-code/AFA_Stream_Framework))

(For the workshop the code is also available [here](https://cloud.ovgu.de/s/gbfnZgbqYYXR9fZ))
    

### 2. Installation Instructions

Create a new project in Pycharm and add the provided code
    
Run following command in console:

    pip install -r requirements.txt

Double click **framework.py** and run it:
--> Should show 4 different plots (performance, budget, running feature importance, violin plots)

## 🗒️ Workshop Modules
### 📖 1. Getting comfortable with the framework

Learn how to change the feature scoring, as well as selection strategies and feature costs.
Run multiple experiments with different degrees of missing data, varying datasets, and
different AFA strategies.

### ⚙️ 2. Adding custom features

Think of a custom feature you might like and add it to the framework.
For example:
- Analysis which features where acquired the most
- Storing history data to compare different scoring mechanisms visually
- Add window-based evaluation
- Add new feature scoring functions e.g. entropy-based from [Yuan et al., 2018](https://dl.acm.org/doi/pdf/10.1145/3167132.3167188?casa_token=VWC1t85My58AAAAA:IUBfQ0nM7QkhKK-out5o1hJx32b4pf7t1v05FdhmXlvO9SuyE8q8qI9DPs28H3ZYoSCpRfKAw3Jl)



## 🎯 Workshop Goals

By the end of this workshop, you will:

Be able to run AFA experiments with different feature costs, different acquisition strategies, and 
feature scoring methods (random and Average Euclidean Distance).
Furthermore, you will know where to change the provided framework to implement custom scoring -, as
well as evaluation metrics, and new feature set selection strategies.

## Papers
- Beyer et al. [Active feature acquisition on data streams under feature drift](https://link.springer.com/content/pdf/10.1007/s12243-020-00775-2.pdf)
- Büttner et al. [Reducing Missingness in a Stream through Cost-Aware Active Feature Acquisition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10032414&casa_token=qKuCvUooETQAAAAA:V1ojaQ93TCN54hb9R2LJU2MEIpR0aOPOWj4FyfYFV0PtIApK2zpMHabw1f1FxNnpNHda4XD_Vw&tag=1)
- Büttner et al. [Joining Imputation and Active Feature Acquisition for Cost Saving on Data Streams with Missing Features](https://link.springer.com/chapter/10.1007/978-3-031-45275-8_21)
