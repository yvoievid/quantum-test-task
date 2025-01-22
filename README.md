# Quantum Test Task

This document outlines the solution to the Quantum test task and provides instructions for testing it.

---

## Prerequisites

- Python 3.7 or higher
- `pip` (Python package manager)

---

## Step 1: Create a Virtual Environment

To ensure a clean and isolated environment for the project, create a virtual environment using `venv`:

### Command:

```bash
python3 -m venv test_task_env
```

Or, if `python` points to Python 3 on your system:

```bash
python -m venv test_task_env
```

---

## Step 2: Activate the Virtual Environment

### On Linux/MacOS:

```bash
source test_task_env/bin/activate
```

### On Windows:

```bash
test_task_env\Scripts\activate
```

Once activated, you should see the virtual environment name in your terminal prompt.

### Upgrade `pip`:

```bash
pip install --upgrade pip
```

---

## Step 3: Install Dependencies

Install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Task 1: Counting Islands (Classical Algorithms)

Explore the algorithm and its implementation in the `task_1_islands` Jupyter notebook.

---

## Task 2: Regression on Tabular Data (General Machine Learning)

The solution includes:

- Training and prediction scripts in the `task_2_classical_ml` folder.
- Exploratory Data Analysis (EDA) in the `eda.ipynb` file.

### Model Details:

- **Algorithm**: XGBoost (chosen for its superior RMSE across all tested models).

### To train the model:

```bash
python train.py
```

### To generate predictions:

```bash
python predict.py
```

All predictions, along with the training data, are stored in the `task_2_classical_ml/data` folder.

---

## Task 3: MNIST Classifier (OOP)

### Installation:

To test the third task, install it as a module using the following command:

```bash
cd ./task_3_mnist
pip install -e .
```

### Testing:

You can find the testing details in the `task_3_experiment.ipynb` file.

---

## Author

**Yurii Voievidka**

---

### Thank you for reviewing this solution! If you have any questions or feedback, feel free to reach out.

