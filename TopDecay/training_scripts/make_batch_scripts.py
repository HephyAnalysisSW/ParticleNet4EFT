
import os
import pathlib
import argparse
import pathlib
import yaml
import datetime

parser = argparse.ArgumentParser()

BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "batch_config.yaml"
DEFAULT_OUT_PATH = BASE_DIR / datetime.date.today().isoformat()

parser.add_argument(
    "-c",
    "--config-file",
    default=DEFAULT_CONFIG,
)

parser.add_argument(
    "-o",
    "--out-path",
    type=pathlib.Path,
    default=DEFAULT_OUT_PATH,
)

parser.add_argument(
    "--type",
    choices=["sh", "sbatch"],
    default="sbatch",
)

parser.add_argument(
    "--submit",
    action="store_true",
)


args = parser.parse_args()

with open(args.config_file, "r") as c:
    config = yaml.safe_load(c)

print(config)

# get the sbatch commands
sbatch_args=config["sbatch_args"]
sbatch_commands = '''#!/bin/bash
#

'''
for name, value in sbatch_args.items():
    sbatch_commands+=f"#SBATCH {name} {value}{os.linesep}"
sbatch_commands+=f"{os.linesep*3}"

# check equal length of network_configs_var entries, extend network_configs_const to that len
n_network_configs = 0
for arg1, values1 in config["network_configs_var"].items():
    for arg2, values2 in config["network_configs_var"].items():
        assert len(values1) == len(values2), f"{arg1} and  {arg2} are not equal length."
        n_network_configs = len(values1)
network_configs_var = config["network_configs_var"]
network_configs_const = config["network_configs_const"]
for name, value in network_configs_const.items():
    value *= n_network_configs

# get model names from model prefix
model_names = []
for model_prefix in config["network_configs_var"]["--model-prefix"]:
    model_names.append(model_prefix.split("/")[-2])

# make out dir
pathlib.Path.mkdir(args.out_path, parents=True, exist_ok=True)

# write script files
for i in range(n_network_configs):
    train_py = f"train.py \\{os.linesep}"

    for name, value in network_configs_var.items():
        train_py += f"{name} {value[i]} \\{os.linesep}"

    for name, value in network_configs_const.items():
        train_py += f"{name} {value[i]} \\{os.linesep}"

    script_file = args.out_path / f"{model_names[i]}.{args.type}"
    with open(script_file, "w") as file:
        # sbatch_commands
        file.write(sbatch_commands)
        file.write(train_py)
        # file.seek(-1,2)
        # file.truncate()

    if args.type == "sh":
        os.system(f"chmod +x {script_file}")


if args.submit:
    os.system(
        f"for script in {args.out_path}/*.sbatch; do echo $script; sbatch $script; done"
    )