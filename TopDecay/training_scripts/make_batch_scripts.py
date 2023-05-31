
import os
import pathlib
import argparse
import pathlib
import yaml
import datetime
import shutil
import glob

parser = argparse.ArgumentParser()

BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_CONFIG = BASE_DIR / "batch_config.yaml"
DEFAULT_OUT_PATH = BASE_DIR / datetime.date.today().isoformat()

parser.add_argument(
    "-c",
    "--config-file",
    type=pathlib.Path,
    default=None,
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

parser.add_argument(
    "--submit-consecutive",
    action="store_true",
)

parser.add_argument(
    "--submit-only",
    action="store_true",
)

parser.add_argument(
    "--copy-config",
    action="store_true",
)


args = parser.parse_args()

if args.config_file is not None:
    config_file=args.config_file
elif not args.submit_only:
    config_file_list = glob.glob(str(args.out_path / "*.yaml"))
    assert len(config_file_list) >= 1, "out path must contain at least one config yaml file"
    if len(config_file_list)==1:
        config_file = config_file_list[0]
    else:
        print("choose from config files")
        for i, file in enumerate(config_file_list):
            print(f"[{i}]: {file}")
        file_idx = int(input())
        config_file = config_file_list[file_idx]
    

if not args.submit_only:
    print(f"using config {config_file}")
    with open(config_file, "r") as c:
        config = yaml.safe_load(c)

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
        if "--tensorboard" in train_options_var:
            train_options_var["--tensorboard"][i] = train_options_var["--tensorboard"][i].replace("auto", auto_name)               

    # get model names from model prefix
    if "script_name" not in config:
        script_names = []
        for model_prefix in config["train_options_var"]["--model-prefix"]:
            script_names.append(model_prefix.split("/")[-2])
    else:
        assert len(script_names:=config["script_name"]) == n_train_scripts, "Number of script names must match number of train options."

    # make out dir
    pathlib.Path.mkdir(args.out_path, parents=True, exist_ok=True)

    # write script files
    for i in range(n_train_scripts):
        train_py = f"python train.py \\{os.linesep}"

        for name, value in train_options_var.items():
            if value[i] is not None:
                train_py += f"{name} {value[i]} \\{os.linesep}"

        for name, value in train_options_const.items():
            train_py += f"{name} {value[i]} \\{os.linesep}"

        for name, value in network_options.items():
            if value[i] is not None:
                train_py += f"--network-option {name} '{value[i]}' \\{os.linesep}"

        # remove the line continuation from the last line of the script
        len_line_cont = len(f" \\{os.linesep}")
        train_py = train_py[:-len_line_cont]

        script_file = args.out_path / f"{script_names[i]}.{args.type}"
        with open(script_file, "w") as file:
            file.write(sbatch_commands)
            file.write(train_py)
            print(f"script written to {script_file}")

        # make executable if .sh
        if args.type == "sh":
            os.system(f"chmod +x {script_file}")




# copy config-file to out-path
if args.copy_config:
    print(f"copying config to {args.out_path / args.config_file.name}")
    shutil.copy(args.config_file, args.out_path)

if (args.submit or args.submit_only):
    os.system(
        f"for script in {args.out_path}/*.{args.type}; do echo $script; sbatch $script; done"
    )

# will submit scripts with the 'afterok' slurm dependency, so they only start after the previous finished succesfully
if args.submit_consecutive:
    sub_string = """
first_iter=0"""

    sub_string += f"""

for script in {args.out_path}/*.{args.type}; do"""

    sub_string += """

if [ $first_iter -eq 0 ]

then

(( first_iter++ ))
job_id=$(sbatch $script)
echo $job_id
echo using $script
job_id=${job_id#Submitted batch job }

else

job_id=$(sbatch -d afterok:$job_id $script)
echo $job_id
echo using $script
job_id=${job_id#Submitted batch job }

fi

done"""
    # print(sub_string)
    os.system(sub_string)