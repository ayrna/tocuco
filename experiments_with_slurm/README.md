# User guide: cluster experimentation framework (Slurm)

Welcome to the research group's execution and experimentation environment. This framework is designed to automate the launch, management, and results collection of Machine Learning and Deep Learning experiments on the Slurm-managed cluster.

The system uses **Submitit** to interact seamlessly with the Slurm queues.

Below are the necessary steps to launch a complete experimentation pipeline using this framework.

---

## 1. Prepare the experiment execution flow

An execution flow (`flow`) defines the main function that our experiment will run. In this project, the main difference between flows is not strictly whether we use classical Machine Learning or Deep Learning, but rather **how and where the hyperparameter search is performed**.

By default, the system supports two main approaches or *pipelines*:

* **`external_cv` (distributed search):** ideal for computationally expensive training (common in Deep Learning). The orchestrator "splits" the search and launches **an independent Slurm job for each parameter configuration**.
* **`internal_cv` (local search):** ideal for fast training (common in classical ML). A single job is sent to Slurm, and the script itself manages the search internally (for example, using Scikit-Learn's `GridSearchCV` or `RandomizedSearchCV`).

In the `execution/flows` folder, you will find two example flows (`dl` focused on `external_cv`, and `ml` focused on `internal_cv`). The core of the system (the orchestrator and workers) is completely independent.

For an execution flow to be usable when launching experiments, it must be registered in the `execution/flows/__init__.py` file, within the `REGISTRY` dictionary:

```python
REGISTRY = {
    "dl": {
        "function": run_dl_flow,
        "supported_pipelines": ["external_cv"],
    },
    "ml": {
        "function": run_ml_flow,
        "supported_pipelines": ["internal_cv"],
    },
}
```

The functions defining the execution flow can take whatever parameters are necessary depending on the experimentation being carried out. The default `dl` flow includes the following:

```python
def run_dl_flow(
    *,
    data_dir,
    dataset,
    n_folds=None,
    val_size=None,
    fold=None,
    results_dir="./results",
    results_totuco_dir="./results_tocuco",
    estimator_name="resnet18classifier",
    estimator_config={},
    batch_size=128,
    seed=0,
    interactive=False,
    n_jobs=1,
    search_n_iter=None,
    use_gpu_if_available=True,
    dry_run=False,
    cv_scoring="amae",
    export_tocuco_results=False,
):
```

### 1.1. Mandatory parameters in flows for the `external_cv` pipeline
For execution flows designed to be used with the distributed parameter search pipeline in Slurm, the orchestrator will pass the following parameters to each experiment when generating the different configurations for each job:
* `dataset`
* `estimator_name`
* `fold`
* `seed`

Therefore, the function defining the execution flow should have at least these parameters and use them to load the specific dataset, the correct estimator, partition the fold if necessary, and use the correct seed.

> **Note:** the orchestrator is only responsible for sending these parameters. It is the responsibility of the user programming the execution flow to use them properly.

### 1.2. Mandatory parameters in flows for the `internal_cv` pipeline
For execution flows designed for the local parameter search pipeline, the parameters received from the orchestrator will be:
* `dataset`
* `estimator_name`
* `seed`

In this case, the `fold` is not received since the parameter search is internal.

### 1.3. Optional parameters in execution flows
In addition to the mandatory parameters indicated in the previous sections, the function defining the execution flow will receive any other configuration parameter present in the experiment's configuration file.

---

## 2. Prepare the experiment configuration

The experiment configuration is entirely based on configuration files, allowing you to keep stored recipes to use whenever necessary. These configurations are located in `execution/config`, with each file representing a different setup. 

The configuration in the `default.py` file will be used by default if no other is specified at runtime. To add additional configurations, simply create new files; the file name (without `.py`) will represent the configuration name.

Additionally, the following configuration files are available:

* `tocuco.py`: for running tocuco experiments.
* `images`: for running image datasets.
* `ml`: basic configuration for launching experiments in the internal pipeline.

Each configuration file is a Python module that defines the different configuration attributes as variables in the module's global scope.

### 2.1. General experiment configuration
There are some attributes that must be present in any configuration, as they define general experimentation settings:

```python
experiment_name = "experiments"
pipeline = "external_cv"  # "external_cv" or "internal_cv"
flow = "dl" # ml - "internal_cv" / dl - "external_cv"
```

Similarly, we must always define the `estimators`, `datasets`, and `seeds` lists, which will determine the names of the estimators, the datasets to be used, and the number of seeds to execute.

### 2.2. Required attributes for the `external_cv` pipeline
When using the external parameter search pipeline, some configuration attributes are essential to define how this distributed search will be conducted:
* `n_folds`: determines the number of folds to be used to cross-validate the parameter configurations. It must be `None` if `val_size` is set.
* `val_size`: determines the size of the validation set (holdout) to be used to cross-validate the parameter configurations. It must be `None` if `n_folds` is set.
* `search_n_iter`: number of iterations for the random parameter search. If the value is larger than the grid size, a grid search will be performed.
* `val_metric`: name of the metric to be used to cross-validate the parameter configurations (must be defined in `experiments/utils/metrics.py`).
* `greater_is_better`: indicates whether the previous metric should be maximized.

### 2.3. Required attributes for the `internal_cv` pipeline
On the other hand, the internal pipeline does not require additional attributes in the configuration file, as the parameter search will be the direct responsibility of the execution flow function itself. However, it is possible to use `n_folds`, `val_size` and `search_n_iter`.

### 2.4. Resource allocation attributes
In the `Resources` section of the configuration file, there are attributes that allow you to specify the resources needed to run each job on the cluster:

```python
############ Resources ############
# Number of CPUs requested for each job
cpus = 2
# Memory in GB for each job
memory = 10
gpus = 1
# Nice level for Slurm jobs (lower is higher priority)
nice = 0
# Select GPU type (according to slurm Gres Type)
# "" empty for any GPU, or "normal_vram" (11GB) / "high_vram" (24GB+)
gpu_type = ""
# gpu_legacy: set to True to use older GPUs too
gpu_legacy = False
# Set to False to disable GPU usage even if available
use_gpu_if_available = True
# Set to True to get structured tocuco results
export_tocuco_results = True
# Max time for each job in HH:MM:SS
max_time = "05:00:00"
# Max concurrent jobs in Slurm
max_concurrent_jobs = 500
```

> **Note:** this only requests these resources from Slurm. Actually utilizing them (configuring estimator internal threads or moving the model to CUDA) is the user's responsibility within their flow.

To execute using a GPU, it will be necessary to set the `gpus` parameter to 1 or more. If you have any specific hardware requirements, you can indicate the type with `gpu_type` (`normal_vram` or `high_vram`). Additionally, you can enable `gpu_legacy` if you allow working with older GPUs (disabled by default to avoid incompatibilities with the environment's CUDA versions).

The `max_time` parameter defines the maximum execution time for each job. Try to adjust it as accurately as possible, as Slurm uses it to assign the job to a short, normal, or long queue, which affects global queue waiting times. If your process exceeds this time, Slurm will forcefully terminate it.

There is also a `Resources override` section within `default.py` that allows overriding these resources for specific datasets or estimators that require more power or memory.

### 2.5. Output generation attributes
In the output directories section, the following attributes exist:
```python
############ Output directories ############
# Remayn results path
results_dir = "./results"
# Experiments execution logs (slurm sh and logs)
logs_output_dir = "./logs"
# Tocuco results paths
results_tocuco_dir = "./results_tocuco_structured"
# Best configurations file path (pickle)
# Path is relative to the experiment log directory
best_configs_file = "best_configs.pkl"
```
* `results_dir`: directory where results will be saved using `remayn`.
* `results_tocuco_dir`: directory where Tocuco's structured results will be saved.
* `logs_output_dir`: submitit logs directory (contains the generated Slurm `.sh` scripts, serialization pickle files, and standard outputs).
* `best_configs_file`: file where the `aggregator` step stores the optimal configuration found, so it can later be read by the final training phase.

### 2.6. Results collection attributes
The `Results Collection Config` section contains attributes related to the generation of the final report. For more details on its behavior, consult the documentation within the `default.py` file itself.

### 2.7. Adding custom attributes to the configuration
You can add any new attribute with whatever name you wish. This attribute will be automatically sent to the execution flow function. The system is flexible: only attributes defined as parameters in your function's signature will be injected, safely ignoring any extras without throwing errors.

---

## 3. Experimentation phases

Experimentation is divided into different sequential phases, which vary depending on the chosen pipeline.

### 3.1. Experimentation phases in the `external_cv` pipeline
It consists of four automated phases:
1.  **Parameter search (paramsearch, `ps`):** launches an independent job to Slurm for each parameter configuration in the grid (or as many as the `search_n_iter` parameter indicates using random search).
2.  **Results aggregation (`aggregator`):** reviews the results of all previous configurations and extracts the optimal one for each estimator, dataset, and seed, saving it in the best configurations pickle file.
3.  **Final training (`final`):** performs a definitive training run without validation using the optimal configuration retrieved by the `aggregator`.
4.  **Results collection (`results_collector`):** collects the final results and generates an Excel file with the metrics defined in `experiments/utils/metrics.py`.

### 3.2. Experimentation phases in the `internal_cv` pipeline
By performing the parameter search within the execution flow itself, it is reduced to two phases:
1.  **Training (`icv`):** launches the configured execution flow so that it performs the training and the internal parameter search in a single Slurm block.
2.  **Results collection (`results_collector`):** collects the results and generates the final Excel report.

---

## 4. Launching the experimentation

To launch the complete experimentation pipeline, the main file `run_experiments.py` is used. 

> **Architecture note (Submitit vs HTCondor):** unlike older systems where one script called another intermediate executable file, this framework uses **Submitit**. The `run_experiments.py` file does not call other physical scripts on the node; it takes your flow function directly in memory, packages it (creates a *pickle*), and asks the Slurm worker to unpackage and run it directly on the assigned node.

The script accepts the following command-line arguments:
* `--config`: indicates the configuration to use (file name in `execution/config` without `.py`). If omitted, `default` will be used.
* `--step`: allows specifying which specific steps to execute (useful for resuming executions or skipping phases). Use `--help` to see valid values (`ps`, `aggregator`, `final`, `results_collector`, `icv`).
* `--dry-run`: performs a simulation. It launches the jobs in Slurm but with the instruction not to execute the heavy training. For this to work, your flow must accept the `dry_run=False` parameter and react by terminating immediately or printing the configuration.
* `--clear-logs`: completely deletes the Submitit logs folder before starting (recommended for cleaning up residue from previous tests).

Launch example using the `tocuco` config and clearing old logs:
```bash
python run_experiments.py --config tocuco --clear-logs
```

---

## 5. Debugging experiments (debug mode)

The `debug_dl_flow.py` and `debug_ml_flow.py` files **are not part of the cluster's massive execution pipeline**.

Their purpose is to allow you to test your execution flow (check for syntax errors, ensure data loads correctly, or verify the model doesn't throw errors) quickly and directly, without activating the Submitit orchestration infrastructure.

You can modify these files freely by changing the parameters directly inside their code (in the `config = dict(...)` dictionary of the `main` method). 

They can be run locally on your machine:
```bash
python debug_dl_flow.py
```

Or, recommended for reserving the necessary resources for execution, via a short interactive Slurm allocation using `srun`:
```bash
srun -p normal_gpu --gres=gpu:1 --pty python debug_dl_flow.py
```

---

# Quick guide to basic slurm commands

Use these essential commands to monitor cluster status, control your processes, and perform interactive debugging.

### 1. Cluster status and resources
* **`sinfo`**: shows the general status of the cluster nodes (`idle`, `alloc`, `mix`, `down`) and available partitions.
* **`sshare -U`**: allows you to check your *Fair-share* status and verify your current priority based on the resources you have consumed recently.
* **`scontrol show node <node_name>`**: shows the detailed configuration of a physical node (how many CPUs, GPUs, and memory are free or occupied in real-time).

### 2. Job monitoring
* **`squeue -u $USER`**: shows a filtered view of **only your jobs** in the execution queue, avoiding cluttering the screen with other users' experiments.
* **`scontrol show job <job_id>`**: shows detailed information about a job (active or recently finished). It's vital for discovering why a job is in a pending (`PD`) state.
* **`sacct`**: check the history of your completed or running jobs. Recommended usage for auditing real RAM consumption:
    ```bash
    sacct -j <job_id> --format=JobID,JobName,State,ExitCode,MaxRSS
    ```

### 3. Job control and cancellation
* **`scancel <job_id>`**: cancels and stops a job or an entire job array.
* **`scancel <job_id>_<task_id>`**: cancels a specific task (a specific index) within a Job Array generated by Submitit, without stopping the other tasks of the experiment.
* **`scancel -u $USER`**: deletes absolutely all your active and pending jobs from the cluster ("panic button").

### 4. Development and advanced interactivity
* **`salloc`**: requests a real-time resource allocation from the cluster and keeps your terminal waiting for them to be granted.
* **`srun --pty <executable>`**: launches a process cloning your terminal's input/output inside a cluster node.

**Recommended interactive flow for deep debugging:**
To enter a worker node and test code directly on a real GPU, run this combination:
```bash
salloc --partition=gpu_short --gres=gpu:1 srun --pty bash
```
Once the cluster grants you the node and you enter its internal terminal, you can launch your interactive environment or debugging scripts:
```bash
python debug_dl_flow.py
```
*(To exit the node and release the cluster resources immediately, type `exit` in the terminal).*