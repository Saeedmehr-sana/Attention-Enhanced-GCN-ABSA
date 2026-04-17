import subprocess
import re

def run_experiments(models, datasets, seeds):
    results = {}  # Store results for each combination of model, dataset

    for model in models:
        for dataset in datasets:
            test_acc_values = []  # Store test_acc values for averaging
            test_f1_values = []   # Store test_f1 values for averaging

            print(f"Running for model: {model}, dataset: {dataset} with seeds {seeds}...")

            for seed in seeds:
                print(f"  Running with seed {seed}...")

                # Run the command and capture the output
                result = subprocess.run(
                    [
                        "python", "train.py",
                        "--model_name", model,
                        "--dataset", dataset,
                        "--learning_rate", "1e-3",
                        "--seed", str(seed),
                        "--batch_size", "16",
                        "--hidden_dim", "300"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
               
                # Combine stdout and stderr for parsing
                output = result.stdout + result.stderr

                # Extract max_test_acc_avg and max_test_f1_avg using regex
                test_acc_match = re.search(r"max_test_acc_avg:\s*([\d.]+)", output)
                test_f1_match = re.search(r"max_test_f1_avg:\s*([\d.]+)", output)

                if test_acc_match and test_f1_match:
                    test_acc = float(test_acc_match.group(1))
                    test_f1 = float(test_f1_match.group(1))
                    test_acc_values.append(test_acc)
                    test_f1_values.append(test_f1)
                    print(f"    Extracted test_acc: {test_acc}, test_f1: {test_f1}")
                else:
                    print(f"    Could not extract test_acc and test_f1 for seed {seed}.")

            # Calculate the average for this model and dataset
            avg_test_acc = sum(test_acc_values) / len(test_acc_values) if test_acc_values else 0
            avg_test_f1 = sum(test_f1_values) / len(test_f1_values) if test_f1_values else 0

            # Store results for the current model-dataset pair
            results[(model, dataset)] = (avg_test_acc, avg_test_f1)

    # Print results
    print("\nFinal Results (averaged over 3 seeds):")
    for (model, dataset), (avg_test_acc, avg_test_f1) in results.items():
        print(f"Model: {model}, Dataset: {dataset} -> Average test_acc: {avg_test_acc}, Average test_f1: {avg_test_f1}")

    return results

# Specify the models, datasets, and seeds to test
models = [
    "cnn", "lstm", "bi_lstm", "bi_lstm_cnn", "gcn", "bi_lstm_gcn",
    "cnn_aspect", "lstm_aspect", "bi_lstm_aspect", "bi_lstm_cnn_aspect",
    "gcn_aspect", "bi_lstm_gcn_aspect"
]
datasets = ["lap14", "rest14", "rest15", "rest16"]
seeds = [9, 39, 109]

# Run the experiments
run_experiments(models, datasets, seeds)