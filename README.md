# Quantum Test Task 

This is the solution of Quatum test task, please follow the instructions to test it.

## Prerequisites
- Python 3.7 or higher
- `pip` (Python package manager)

---

## Step 1: Create a Virtual Environment with `venv`

Run the following command to create a virtual environment in the desired directory:

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

then 

```bash
pip install --upgrade pip
```

---

## Step 3: Install Dependencies


```bash
pip install -r requirements.txt
```

---

## Tak 1: Counting islands. Classical Algorithms

---

You can find the algoright and testing of it in `task_1_islands` jupyter notebook. 

## Task 2: Regression on the tabular data. General Machine Learning

You can find the train, and predict scripts in `task_2_classical_ml` and Exploritory Data Analysis in file `eda.ipynb`

I have used XGBoost as this ensemble model has the best RMSE across all tested ones.

to train the model run:
```bash
python train.py
```

and run following command to create predictions:
```bash
python predict.py
```
all predictions along with training data are stored in `task_2_classical_ml/data` folder 

---

## Task 3:

to test the Third test task run the following command to install it as module

```bash
cd ./task_3_mnist
pip install -e .
```

After that you can see testing of solution in `task_3_experiment.ipynb`


### Author: Yurii Voievidka