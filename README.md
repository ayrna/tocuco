## TOC-UCO: The Tabular Ordinal Classification repository of the University of Córdoba

:link: The TOC-UCO repository can be downloaded from [this website](https://www.uco.es/grupos/ayrna/tocuco). The following bash command can also be executed:

```bat 
wget https://www.uco.es/grupos/ayrna/datasets/TOC-UCO.zip
```

## :bar_chart: TOC-UCO Datasets Characteristics

The TOC-UCO benchmark contains two main groups of tabular ordinal classification datasets: **Discretised regression datasets** (continuous problems converted to ordinal scales) and **Originally OC datasets** (problems inherently ordinal). 

* $K$ represents the number of input variables.
* $Q$ denotes the number of classes.
* **Class distribution** lists the percentage of patterns belonging to each class (from class 1 to class $Q$).
* **IR** stands for the Imbalance Ratio (values equal to 1 denote a perfectly balanced problem).

### Discretised regression datasets

| Dataset | #Train | #Test | $K$ | $Q$ | Class distribution | IR |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| forestfires | 361 | 156 | 8 | 4 | (0.75 0.13 0.04 0.07) | 3.51 |
| machine | 146 | 63 | 6 | 4 | (0.70 0.14 0.08 0.08) | 2.44 |
| buoysFlux46026 | 4090 | 1754 | 8 | 5 | (0.39 0.24 0.15 0.11 0.10) | 1.36 |
| buoysFlux46059 | 4090 | 1754 | 8 | 5 | (0.45 0.25 0.13 0.08 0.09) | 1.62 |
| census1 | 15,948 | 6836 | 8 | 5 | (0.46 0.24 0.13 0.08 0.09) | 1.63 |
| census2 | 15,948 | 6836 | 16 | 5 | (0.46 0.24 0.13 0.08 0.09) | 1.63 |
| buoysFlux46069 | 4090 | 1754 | 8 | 6 | (0.34 0.24 0.16 0.10 0.07 0.09) | 1.42 |
| calhousing | 14,448 | 6192 | 8 | 7 | (0.31 0.14 0.11 0.15 0.09 0.06 0.07 0.07) | 1.18 |
| buoysHeight46026 | 4090 | 1754 | 8 | 8 | (0.20 0.16 0.16 0.12 0.11 0.09 0.06 0.10) | 1.14 |
| buoysHeight46069 | 4090 | 1754 | 8 | 8 | (0.21 0.18 0.14 0.13 0.11 0.09 0.05 0.09) | 1.20 |
| cancerTreatment | 748 | 321 | 12 | 8 | (0.24 0.19 0.16 0.14 0.10 0.09 0.06 0.07) | 1.31 |
| abalone | 2923 | 1254 | 10 | 9 | (0.20 0.14 0.16 0.15 0.12 0.06 0.05 0.03 0.09) | 1.46 |
| bank1 | 5734 | 2458 | 8 | 9 | (0.29 0.14 0.12 0.10 0.08 0.07 0.06 0.05 0.09) | 1.31 |
| buoysHeight46059 | 4090 | 1754 | 8 | 9 | (0.20 0.16 0.14 0.12 0.10 0.08 0.07 0.05 0.09) | 1.20 |
| computer1 | 5734 | 2458 | 12 | 9 | (0.07 0.02 0.03 0.06 0.08 0.11 0.17 0.21 0.25) | 1.97 |
| computer2 | 5734 | 2458 | 21 | 9 | (0.07 0.02 0.03 0.06 0.08 0.11 0.17 0.21 0.25) | 1.97 |
| housing | 354 | 152 | 13 | 9 | (0.18 0.12 0.19 0.17 0.12 0.07 0.05 0.04 0.06) | 1.37 |
| insurance | 936 | 402 | 9 | 9 | (0.28 0.19 0.16 0.12 0.06 0.05 0.03 0.03 0.08) | 1.75 |
| soybean | 224 | 96 | 9 | 9 | (0.21 0.18 0.13 0.12 0.07 0.07 0.08 0.05 0.08) | 1.24 |
| bank2 | 5734 | 2458 | 32 | 10 | (0.45 0.13 0.09 0.07 0.05 0.04 0.03 0.03 0.03 0.07) | 2.04 |
| cancerDeathRate | 2132 | 915 | 29 | 10 | (0.10 0.09 0.10 0.11 0.12 0.13 0.11 0.08 0.06 0.10) | 1.05 |
| concreteStrength | 721 | 309 | 8 | 10 | (0.14 0.08 0.13 0.14 0.10 0.11 0.08 0.07 0.06 0.09) | 1.08 |
| realState | 289 | 125 | 6 | 10 | (0.10 0.09 0.10 0.09 0.11 0.13 0.11 0.09 0.07 0.11) | 1.03 |
| stock | 665 | 285 | 9 | 10 | (0.14 0.08 0.06 0.12 0.11 0.10 0.09 0.08 0.08 0.13) | 1.08 |

### Originally OC datasets

| Dataset | #Train | #Test | $K$ | $Q$ | Class distribution | IR |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| balanceScale | 437 | 188 | 4 | 3 | (0.46 0.08 0.46) | 2.35 |
| mammoexp | 288 | 124 | 5 | 3 | (0.57 0.25 0.18) | 1.38 |
| newthyroid | 150 | 65 | 5 | 3 | (0.14 0.7 0.16) | 1.96 |
| tae | 105 | 46 | 54 | 3 | (0.32 0.33 0.34) | 1.00 |
| car | 1209 | 519 | 21 | 4 | (0.7 0.22 0.04 0.04) | 4.46 |
| childrenAnemia | 4098 | 1757 | 14 | 4 | (0.28 0.28 0.41 0.03) | 2.88 |
| gymExerciseTracking | 681 | 292 | 17 | 4 | (0.2 0.38 0.31 0.1) | 1.36 |
| heartDisease | 205 | 89 | 13 | 4 | (0.64 0.13 0.09 0.15) | 1.97 |
| LESTSensors | 3578 | 1534 | 6 | 4 | (0.16 0.14 0.23 0.46) | 1.30 |
| LEVXSensors | 3578 | 1534 | 6 | 4 | (0.33 0.08 0.1 0.48) | 1.93 |
| problematicInternet | 1484 | 636 | 32 | 4 | (0.59 0.27 0.13 0.01) | 7.55 |
| support | 511 | 219 | 19 | 4 | (0.39 0.13 0.08 0.4) | 1.79 |
| swd | 700 | 300 | 10 | 4 | (0.03 0.35 0.4 0.22) | 3.10 |
| eucalyptus | 515 | 221 | 91 | 5 | (0.24 0.15 0.18 0.29 0.14) | 1.10 |
| lev | 700 | 300 | 4 | 5 | (0.09 0.28 0.4 0.2 0.03) | 2.70 |
| nhanes | 3656 | 1567 | 30 | 5 | (0.11 0.28 0.4 0.18 0.04) | 2.15 |
| vlbw | 120 | 52 | 19 | 5 | (0.15 0.17 0.15 0.28 0.25) | 1.10 |
| winequalityRed | 1119 | 480 | 11 | 5 | (0.04 0.43 0.4 0.12 0.01) | 6.11 |
| esl | 341 | 147 | 4 | 6 | (0.11 0.22 0.24 0.28 0.13 0.05) | 1.51 |
| studentPerformance | 463 | 199 | 43 | 8 | (0.07 0.04 0.11 0.24 0.22 0.16 0.11 0.06) | 1.50 |
| era | 700 | 300 | 4 | 9 | (0.09 0.14 0.18 0.17 0.16 0.12 0.09 0.03 0.02) | 1.86 |
| melbourneAirbnb | 14025 | 6011 | 48 | 10 | (0.08 0.09 0.1 0.09 0.05 0.12 0.11 0.09 0.09 0.18) | 1.12 |

---

## Repository Structure (`tocuco_results.zip`)

Once you download and decompress the `tocuco_results.zip` file from [this download link](https://ayrna-nc.ext.uco.es/index.php/s/4gTHXi9FJtZaTWS), you will find a standardised directory structure containing the comprehensive experimental results. The folders and files are organized as follows:

```text
/results_tocuco
|
|---> /[model_name]
      |
      |---> /[dataset_name]
            |
            |---> /predictions_by_seed
            |     |
            |     |---> /seed_0
            |     |     |---> train_predictions.csv
            |     |     |---> test_predictions.csv
            |     |     |---> train_confusion_matrix.txt
            |     |     |---> test_confusion_matrix.txt
            |     |
            |     |---> /seed_1
            |     |     |...
            |
            |---> hyperparameter_configuration.csv
```

### :orange_book: Processing Experimental Results

To facilitate the exploration, parsing, and analysis of the data contained within `tocuco_results.zip`, a dedicated Jupyter Notebook named `results_processor.ipynb` is provided in this repository, located in the `results` folder.

This notebook automates the process of reading the individual seed predictions, reconstructing metrics across all datasets, and compiling final performance tables without manual overhead.

### Detailed File Specifications

#### 1. Individual Seed Predictions (`train_predictions.csv` & `test_predictions.csv`)
These files map specific pattern indices back to their original array configurations via `train_masks.json`.

* **`train_predictions.csv`**: Extracts indices where the mask array evaluation is `True` for that specific dataset/seed combination (e.g., `'dr09_housing_seed_0'`).
* **`test_predictions.csv`**: Extracts indices where the mask array evaluation is `False`.

**Column Structure:**
* **`Pattern ID`**: The exact numerical index (0-based position) of the item inside the original dataset array.
* **`Target`**: The true ground-truth ordinal class expected for the given pattern.
* **`Prediction probabilities`** *(Conditional)*: If the chosen model outputs probabilities, this column must contain an array bounded by `[` and `]` containing the classification probability distribution across all classes. The elements within the array must sum to exactly 1.0.
* **`Prediction`**: The final predicted discrete class. If probabilities are present, this value must equal the `argmax` index of the probability array. If the model does not yield probabilities, this will contain the direct discrete output.

#### 2. Local Confusion Matrices (`train_confusion_matrix.txt` & `test_confusion_matrix.txt`)
Instead of appending globally across a single dataset directory, each individual seed subdirectory will encapsulate its own local evaluation matrices. The string formatting must look as follows:

```text
Seed 0
=====================
[[TN, FP],
 [FN, TP]]
```

### 3. Cross-Validation Log (`hyperparameter_configuration.csv`)
Stored directly inside the root of the dataset folder, tracking the configuration parameters chosen across every executed run.

**Format:**
* **First Column:** `Seed` (Integer identifying the run index).
* **Subsequent Columns:** Dynamically adjusted based on the checked hyperparameters for the given model technique.

### :gear: Hyperparameter Cross-Validation Reference

The following table details the exact hyperparameter grid values evaluated during the cross-validation process for each classification method:

| Method | Hyperparameter | Values |
| :--- | :--- | :--- |
| **Ridge, LogAT, and LogIT** | Regularisation strength | {$10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1, 10^2, 10^3$} |
| | Maximum iterations | {1000, 1500, 3000, 5000} |
| **Ridge** | Intercept inclusion | {True, False} |
| **XGB** | Max depth | {3, 5, 8} |
| | #Estimators | {100, 250, 500, 1000} |
| | Learning rate | {0.01, 0.05, 0.10} |
| | Pattern subsample (%) | {0.75, 0.95, 1.0} |
| | Feature subsample (%) | {0.75, 0.95, 1.0} |
| **MLP, MLP-CLM, and MLP-T** | #Hidden units | {5, 8, 10, 15, 20, 50, 100} |
| | #Epochs | {1000, 1500, 3000, 5000} |
| | Learning rate | {$10^{-5}, 10^{-4}, 10^{-3}$} |
| **MLP-CLM** | Minimum distance | {0.0, 0.1, 0.2} |
| **MLP-T** | Alpha | {0.05, 0.10} |
| **OEAB** | Estimator max depth | {4, 16, 32, 64} |
| | #Estimators ($M$) | {10, 30} |
| | Regularisation strenght | {$10^{-4}, 10^{-3}, 10^{-2}$} |

---

### :search: Hyperparameter Search Strategy

To find the optimal configurations from the grids above, the selection across all methods is strictly optimized based on the **AMAE (Average Absolute Error)** metric, using two different validation strategies depending on the model type:

* **GridSearch**: Evaluates all possible combinations within the defined parameter grid.
* **RandomizedSearch**: Randomly samples a subset of the grid, executing **exactly 20 iterations**.

The search mapping and optimization criteria per method are distributed as follows:

| Method | Optimization Strategy | Performance Metric |
| :--- | :--- | :--- |
| **Ridge** | GridSearch | AMAE |
| **LogAT** | GridSearch | AMAE |
| **LogIT** | GridSearch | AMAE |
| **XGB** | RandomizedSearch | AMAE |
| **MLP** | RandomizedSearch | AMAE |
| **MLP-CLM** | RandomizedSearch | AMAE |
| **MLP-T (MLP-Triangular)** | RandomizedSearch | AMAE |
| **OEAB** | GridSearch | AMAE |

---

### :computer: Related Resources

More about ordinal classification can be found in the [dlordinal package](https://github.com/ayrna/dlordinal), where several ordinal techniques are implemented, together with the most popular ordinal performance metrics.

---

## :books: Citation

If you use this repository or framework in your research, please appreciate it by citing the following work:

```bibtex
@article{Ayllon2026TOC,
    author = {Ayll{\' o}n-Gavil{\' a}n, Rafael and Guijo-Rubio, David and G{\' o}mez-Orellana, Antonio Manuel and B{\' e}rchez-Moreno, Francisco and Vargas-Yun, V{\' i}ctor Manuel and Guti{\' e}rrez, Pedro Antonio},
    journal = {Neurocomputing},
    doi = {10.1016/j.neucom.2026.133528},
    year = {2026},
    pages = {133528},
    title = {TOC-{UCO}: a comprehensive repository of tabular ordinal classification datasets},
    url = {[https://www.sciencedirect.com/science/article/pii/S0925231226009252](https://www.sciencedirect.com/science/article/pii/S0925231226009252)},
    howpublished = {[https://www.sciencedirect.com/science/article/pii/S0925231226009252](https://www.sciencedirect.com/science/article/pii/S0925231226009252)},
}
```