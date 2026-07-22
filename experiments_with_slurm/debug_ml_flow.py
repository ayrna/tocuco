def main():
    import json

    from execution.flows import run_ml_flow

    config = dict(
        data_dir="/mnt/datasets",
        dataset="tocuco_dr04_machine",
        n_folds=3,
        val_size=None,
        results_dir="./results",
        estimator_name="MLP",
        seed=0,
        interactive=True,
        n_jobs=3,
    )

    print(f"Running experiment with config: ")
    print(json.dumps(config, indent=4))

    run_ml_flow(**config)


if __name__ == "__main__":
    main()
