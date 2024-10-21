# End-to-End Chest Cancer Classification Using MLFLOW and DVC

This project aims to predict **adenocarcinoma chest cancer**, one of the most common types of lung cancer, using machine learning techniques. By leveraging this classification model, we aim to provide a mechanism for early detection, potentially identifying related types of chest cancers, such as squamous cell carcinoma and large cell carcinoma. The end goal is to help in the timely detection of these cancers and improve patient outcomes.

The project uses **MLFlow** for experiment tracking, ensuring every model, parameter, and metric is logged for analysis, and **DVC** (Data Version Control) for data pipeline management and versioning. This setup guarantees that the entire workflow is reproducible and scalable.

---

## Project Structure

Here’s an overview of the folder structure:

```
End-to-End-Chest-cancer-Classification-Using-MLFLOW-DVC/
│
├── .github/workflow/                      # CI/CD configurations using GitHub Actions
│   └── .gitkeep                           # Ensures folder is tracked by Git
├── config/
│   └── config.yaml                        # Centralized configuration file
├── params.yaml                            # Parameter configuration (model hyperparameters, paths, etc.)
├── dvc.yaml                               # DVC pipeline stages
├── main.py                                # Entry point to trigger the full pipeline
├── research/
│   └── trials.ipynb                       # Notebook for experiments and trials
├── requirements.txt                       # Required Python libraries for the project
├── setup.py                               # Setup script for installing the project as a package
├── src/                                   # Source code for the project
│   └── ChestCancerClassifier/             # Main project package
│       ├── __init__.py                    # Marks ChestCancerClassifier as a Python package
│       ├── components/                    # ML pipeline components (data ingestion, training, evaluation)
│       │   └── __init__.py                
│       ├── config/                        # Configuration management
│       │   ├── __init__.py
│       │   └── configuration.py           # Handles config loading and management
│       ├── constants/                     # Define constants such as paths
│       │   └── __init__.py
│       ├── entity/                        # Define entities (data models, etc.)
│       │   └── __init__.py
│       ├── pipeline/                      # Pipeline orchestration logic
│       │   └── __init__.py
│       └── utils/                         # Utility functions for common tasks
│           ├── __init__.py
│           └── common.py                  # Helper methods for preprocessing, logging, etc.
└── README.md                              # This file
```

---

## Workflow Overview

The workflow consists of several steps to ensure the smooth development and deployment of the model. Here's an in-depth explanation of each step:

### 1. **Update `config.yaml`**

The `config/config.yaml` file serves as the centralized configuration hub. You should update this file to reflect all the necessary paths for your data, model directories, and other related settings like logging and evaluation criteria.

### 2. **Update `params.yaml`**

`params.yaml` holds the hyperparameters and model settings such as learning rate, batch size, number of epochs, etc. This is where you define key variables that impact model training and evaluation. 

### 3. **Update the Entity**

In the **entity** layer (located in `src/ChestCancerClassifier/entity/`), you define the input-output structure of your data. This step involves updating the data models and structures that flow through the pipeline. For instance, you may define data objects that represent a cancer image and its classification label.

### 4. **Update the Configuration Manager in `src/config`**

In this step, you'll modify the `configuration.py` file under `src/ChestCancerClassifier/config/`. This file is responsible for reading and managing the settings defined in `config.yaml` and `params.yaml`. It simplifies how other components of the pipeline interact with configuration values, keeping your pipeline modular and easy to update.

### 5. **Update the Components**

The components folder (`src/ChestCancerClassifier/components/`) contains the core steps of the ML pipeline:

- **Data ingestion**: Loading and preprocessing images.
- **Model training**: Applying machine learning algorithms for training.
- **Evaluation**: Assessing the performance of the trained model.

Here, you'll update the Python files to reflect the specifics of your project, such as what kind of data you're working with (X-ray images, CT scans) and which algorithms you're using (e.g., CNN, Random Forest).

### 6. **Update the Pipeline**

In the `src/ChestCancerClassifier/pipeline/`, you update the pipeline logic that ties all components together. This step ensures that each component is executed in the correct order — data ingestion, preprocessing, model training, and evaluation. This pipeline manages dependencies between these steps and ensures data flows smoothly from one stage to another.

### 7. **Update `main.py`**

`main.py` is the primary file that executes the entire pipeline. This is where you'll import and run the defined components and initiate the MLFlow tracking. Ensure that the components are integrated properly and that results are logged using MLFlow.

### 8. **Update `dvc.yaml`**

The `dvc.yaml` file outlines the stages in the DVC pipeline. You'll update this file to define the steps your project follows, such as:

- **stages**:
  - data preparation
  - model training
  - evaluation
  - testing

Each stage in `dvc.yaml` depends on outputs from previous stages. This ensures that your pipeline only re-runs steps when their dependencies change.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Lavishgangwani/End-to-End-Chest-cancer-Classification-Using-MLFLOW-DVC.git
cd End-to-End-Chest-cancer-Classification-Using-MLFLOW-DVC
```

### 2. Install the Requirements

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure the Project

Update the **config/config.yaml** and **params.yaml** files with relevant paths, hyperparameters, and other necessary configurations.

### 4. Set Up DVC

To ensure reproducibility and version control for your data, initialize DVC:

```bash
dvc init
```

Then pull the datasets (if stored remotely):

```bash
dvc pull
```

### 5. Running the Pipeline

You can now run the entire pipeline by executing:

```bash
python main.py
```

This will trigger the steps defined in `dvc.yaml`, which include:

- **Data ingestion**
- **Data preprocessing**
- **Model training**
- **Evaluation**
- **Logging results with MLFlow**

### 6. Track Experiments with MLFlow

Ensure that MLFlow is installed and run it in your terminal to launch the MLFlow UI:

```bash
mlflow ui
```

By default, this will open the MLFlow UI on `http://127.0.0.1:5000/`, where you can track experiment metrics, parameters, and model artifacts.

### 7. Reproducing the Results

Since the project uses DVC, you can reproduce any experiment run by executing:

```bash
dvc repro
```

This will ensure that all steps (data processing, model training, etc.) are executed according to the pipeline configuration.

---

## Folder Details

### `config/config.yaml`

This YAML file holds the centralized configuration for different parts of the pipeline. You'll find sections for data paths, model parameters, and logging configurations.

### `params.yaml`

This file contains the hyperparameters for model training, such as the learning rate, batch size, and epochs. By centralizing parameters, we can easily adjust them without modifying the source code directly.

### `src/ChestCancerClassifier/components`

This folder contains all the building blocks of the ML pipeline, including:

- **Data ingestion**: Loading and splitting data.
- **Model training**: Training the machine learning models.
- **Evaluation**: Evaluating model performance.

### `dvc.yaml`

Defines the stages of the DVC pipeline. Each stage corresponds to one step in the ML workflow, from data ingestion to evaluation. DVC ensures that each step is cached and only re-run if the input to a stage changes.

### `main.py`

This is the entry point to the entire project pipeline. When executed, it reads the configuration and parameters, triggers the pipeline, and tracks results via MLFlow.

---

### Acknowledgments

- **MLFlow**: For experiment tracking.
- **DVC**: For data version control and pipeline orchestration.
- **Open-source Libraries**: Including tensorflow, Pandas, and others.

--- 

Happy coding! 😊