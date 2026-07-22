##########################
### EXPERIMENTS CONFIG ###
##########################

experiment_name = "experiments_images"
pipeline = "internal_cv"  # "external_cv" or "internal_cv"
flow = "ml" # ml - "internal_cv" / dl - "external_cv"

############ Experiment settings ############
estimators = [
        ####  Classifiers  ####
        # "resnet18classifier",
        # "resnet18clmclassifier",
        # "resnet18betaclassifier",
        # "resnet18clmwkclassifier",
        # "resnet18clmbetaclassifier",
        # "resnet18clmwkbetaclassifier",
        # "mlpclassifier",
        # "mlpmanualclassifier",
        # "randomforestclassifier",
        # "ridgeclassifier",
        # "xgboostclassifier",

        ####  Ordinal classifiers  ####

        # "resnet18mceloss",
        # "resnet18mcewkloss",
        # "resnet18cdwce",
         "resnet18sord",
        # "resnet18slace",
        # "mlpsordclassifier",
        # "mlpslaceclassifier",
        # "oeabclassifier",
        # "logatcwclassifier",
        # "xgboostclassifierslace",
        # "logitcwclassifier",

        ####   Ordinal classifiers (soft labelling)   ####

        # "resnet18triangular",
        # "resnet18beta_sl",
        # "resnet18exponential",
        # "mlptriangularclassifier",

        ####   Ordinal classifiers (thresholds)   ####

        "resnet18clmclassifier_oc",
        # "resnet18clmwkclassifier_oc",
        # "resnet18clmbetaclassifier_oc",
        # "resnet18clmexponentialclassifier",
        # "resnet18clmtriangularclassifier",
        # "mlpclmclassifier",
        # "mlpclmsordclassifier",
        # "mlpclmslaceclassifier",

    ]
data_dir = "/mnt/datasets"
# datasets must be registered in data_dir / datasets.json
datasets = [
    # "adience",
    "fgnet",
    # "wiki6",
    # "utkface12",
    # "benelli",
    # "alzheimer",
    # "ava",
    # "retinopathy",
    # "smear",
    # "hands_color",
    # "limuc",
    # "knee_kaggle",
    # "AFAD",
    # "csaw-m",
    # "knee_jesi_cppd",
    # "knee_jesi_kl",
    # "CIFAR10",
    # "tiny_adience",
    # "tiny_utkface12",
    # "tiny_wiki6",
    # "tiny_alzheimer",
    # "tiny_retinopathy",
    # "tiny_ava",
]
seeds = 2
n_folds = None # 3
val_size = 0.3 # 0
search_n_iter = 2 # 20
val_metric = "AMAE"
greater_is_better = False
batch_size = 999999
# Number of jobs (used within the experiment)
# It does not affect the resources requested for each job,
# which are defined in the Resources section below
n_jobs = 1
# Whether to perform a dry run (jobs only print the configuration)
# for testing purposes
# Can be overriden with the --dry-run argument
dry_run = False

############ Resources ############
# Number of CPUs requested for each job
cpus = 3
# Memory in GB for each job
memory = 3
gpus = 1
# Nice level for Slurm jobs (lower is higher priority)
# Cannot be negative
nice = 0
# Select GPU type (according to slurm Gres Type)
# "" empty for any GPU, or "normal_vram" (11GB) / "high_vram" (24GB+)
gpu_type = ""
# gpu_legacy: set to True to use older GPUs too
gpu_legacy = False
# Set to False to disable GPU usage even if available
use_gpu_if_available = True
# Max time for each job in HH:MM:SS
max_time = "05:00:00"
# Max concurrent jobs in Slurm
max_concurrent_jobs = 500

############ Resources override ############
# Memory, gpu_type, batch_size and max_time can be overridden
# for specific estimators / datasets using the override dictionaries below
# The key of the dictionary is the estimator name.
# The value is another dictionary where the key is the dataset name
# and the value is the overridden resource value.
# Use * as wildcard for all estimators / datasets
# Example:
# memory_override = {
#     "estimator1" : {
#         "*" : 10,  # Override memory to 10GB for estimator1 on all datasets
#     }
# }
memory_override = {}
gpu_type_override = {}
batch_size_override = {}
max_time_override = {}

############ Output directories ############
# Remayn results path
results_dir = "./results_nuevos"
# Experiments execution logs (slurm sh and logs)
logs_output_dir = "./logs"
# Best configurations file path (pickle)
# This file will be created by the aggregator worker
# Path is relative to the experiment log directory
best_configs_file = "best_configs.pkl"


#################################
### RESULTS COLLECTION CONFIG ###
#################################

# Output path for the collected results (Excel and zip)
prepared_results_dir = "./results_images"
# Appendix for the collected results output file name (Excel and zip)
prepared_results_appendix = "dl"
# Methods to include in the collected results (None for all)
collect_methods = None
# Datasets to include in the collected results (None for all)
collect_datasets = None
# Seeds to include in the collected results (None for all)
collect_seeds = None
# Config fields from the experiment config that will be included in
# the collected results dataframe as columns (None for default)
config_columns_to_include = [
    "estimator_name",
    "dataset",
    "rs",
    "estimator_config.estimator__max_iter",
]
# Best params fields from the experiment config that will be included in
# the collected results dataframe as columns (None for default)
best_params_columns_to_include = [
    "hidden_units",
    "learning_rate",
]
# Whether to include training metrics in the collected results
collect_train = True
# Whether to include validation metrics in the collected results
collect_val = False
# Whether to skip creating the zip file with the collected results
skip_zip = False
# Number of parallel jobs to use for results collection
collect_n_jobs = 4


# NOTE: Additional configuration parameters can be added as needed, and they will be
# automatically passed to the load_and_run_experiment function.
