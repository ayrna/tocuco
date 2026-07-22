def main():
    import json

    from execution.flows import run_dl_flow

    config = dict(
        data_dir="/home/fberchez/DeepLearning/tocuco/",
        dataset="tocuco_dr06_buoysFlux46069",
        n_folds=None,
        fold=None,
        val_size=0.0,
        results_dir="./results",
        estimator_name="logitcwclassifier",
        estimator_config=None,
        batch_size=128,
        seed=0,
        interactive=True,
        n_jobs=3,
        use_gpu_if_available=True,
    )

    print(f"Running experiment with config: ")
    print(json.dumps(config, indent=4))

    run_dl_flow(**config)


if __name__ == "__main__":
    main()
