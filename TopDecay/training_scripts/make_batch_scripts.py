
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

# print(config)

# get the sbatch commands
sbatch_args=config["sbatch_args"]
sbatch_commands = '''#!/bin/bash
#

'''
for name, value in sbatch_args.items():
    sbatch_commands+=f"#SBATCH {name} {value}{os.linesep}"
sbatch_commands+=f"{os.linesep*3}"

# check equal length of train_options_var entries, extend train_options_const to that len
n_train_scripts = 0
for arg1, values1 in config["train_options_var"].items():
    for arg2, values2 in config["train_options_var"].items():
        assert len(values1) == len(values2), f"{arg1} and  {arg2} are not equal length."
        n_train_scripts = len(values1)

train_options_var = config["train_options_var"]
train_options_const = config["train_options_const"]
for name, value in train_options_const.items():
    value *= n_train_scripts
network_options=config["network_options"]

# auto model prefix
if "--model-prefix" not in train_options_var:
    train_options_var["--model-prefix"] = ["models/auto/model"]*n_train_scripts
for i in range(n_train_scripts):
    auto_name = ""
    for name, value in network_options.items():
        auto_name += f"{name}_{value[i]}_"
    for char in "[", "]", "(", ")", " ", ".":
        auto_name = auto_name.replace(char,"")
    auto_name = auto_name.replace(",","_")
    auto_name = auto_name[:-1]
    train_options_var["--model-prefix"][i] = train_options_var["--model-prefix"][i].replace("auto", auto_name)        

# get model names from model prefix
model_names = []
for model_prefix in config["train_options_var"]["--model-prefix"]:
    model_names.append(model_prefix.split("/")[-2])

# make out dir
pathlib.Path.mkdir(args.out_path, parents=True, exist_ok=True)

# write script files
for i in range(n_train_scripts):
    train_py = f"python train.py \\{os.linesep}"

    for name, value in train_options_var.items():
        train_py += f"{name} {value[i]} \\{os.linesep}"

    for name, value in train_options_const.items():
        train_py += f"{name} {value[i]} \\{os.linesep}"

    for name, value in network_options.items():
        train_py += f"--network-option {name} '{value[i]}' \\{os.linesep}"

    # remove the line continuation from the last line of the script
    len_line_cont = len(f" \\{os.linesep}")
    train_py = train_py[:-len_line_cont]

    script_file = args.out_path / f"{model_names[i]}.{args.type}"
    with open(script_file, "w") as file:
        file.write(sbatch_commands)
        file.write(train_py)

    # make executable if .sh
    if args.type == "sh":
        os.system(f"chmod +x {script_file}")


if args.submit:
    os.system(
        f"for script in {args.out_path}/*.sbatch; do echo $script; sbatch $script; done"
    )