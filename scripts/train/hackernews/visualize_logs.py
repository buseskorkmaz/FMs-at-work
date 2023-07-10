import json
import matplotlib.pyplot as plt
import re

# Initialize an empty dictionary to store metrics
metrics = {
    "token_loss": [],
    "v_loss": [],
    "q_loss": [],
    "cql_loss": [],
    "dm_loss": [],
    "iteration": [],
    "epoch": []
    # Add other metrics as needed
}
pattern = r'\<wandb\.sdk\.data_types\.histogram\.Histogram.*?\>'

# Open the log file and read its contents
with open("/u/buseskorkmaz/.lsf/cccCluster/943525.stdout", "r") as file:
    for line in file:
        # Skip lines not starting with "{'eval':"
        if not line.strip().startswith("{'eval'"):
            print(line)
            continue

        # Replace single quotes with double quotes
        line = line.replace("'", '"')
        # Remove non-JSON value "<wandb.sdk.data_types.histogram.Histogram object at 0x151e5c12dea0>"
        line = re.sub(pattern, 'null', line)

        # Try to parse the line as JSON and skip it if an error occurs
        try:
            log_entry = json.loads(line)
        except json.JSONDecodeError:
            continue


        print(log_entry)
        # If the log entry starts with "eval", extract the metrics
        for key in metrics.keys():
            if key in log_entry["eval"]:
                metrics[key].append(log_entry["eval"][key])
        metrics["iteration"].append(log_entry["iteration"])
        metrics["epoch"].append(log_entry["epoch"])

# Plot the metrics
for metric, values in metrics.items():
    if metric not in ["iteration", "epoch"]:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["iteration"], values)
        plt.title(f"{metric} over iterations")
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.show()
