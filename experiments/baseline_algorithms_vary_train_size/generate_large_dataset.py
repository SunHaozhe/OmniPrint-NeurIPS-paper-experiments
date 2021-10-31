"""
Helper module to generate several multilingual datasets
"""

import os 
import glob
import datetime 
import subprocess 
import argparse 
import numpy as np 



ignore_list = ["__module__", "__dict__", "__weakref__", "__doc__"] 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200, help="images per class")
    parser.add_argument("--nb_processes", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="meta1", 
        help="""OmniPrint-meta[X] style MetaDL dataset""")
    return parser.parse_args()


def generate_one_subdataset(parameters, working_dir):
    cmd = "python3 run.py"
    for key, value in parameters.items():
        if key in ignore_list:
            continue 
        if len(key) >= 2 and key[-1] == "_" and key[-2] != "_":
            key = key[:-1]
        if isinstance(value, bool) and value:
            cmd += " --{}".format(key)
        else:
            cmd += " --{} {}".format(key, str(value))

    # run the command 
    subprocess.call(cmd.split(), cwd=working_dir)

def generate_one_multilingual_dataset(args, parameters, idx):
    if args.nb_processes is not None:
        parameters["nb_processes"] = args.nb_processes

    working_dir = os.path.join(os.pardir, "omniprint")
    
    alphabets = []
    for path in sorted(glob.glob(os.path.join(working_dir, args.text, "*.txt"))):
        alphabets.append(path)

    dataset_name = "large_meta3"
    
    parameters["output_dir"] = os.path.join(os.pardir, "baseline_algorithms_vary_train_size", dataset_name)
    for i, alphabet in enumerate(alphabets):
        alphabet_basename = os.path.splitext(os.path.basename(alphabet))[0]
        parameters["dict_"] = alphabet
        parameters["output_subdir"] = "{}".format(alphabet_basename)
        generate_one_subdataset(parameters, working_dir)


if __name__ == "__main__":
    args = parse_arguments()
    args.text = "alphabets/omniglot_like"

    params_list = []


    pre_elastic = 0.035
    margins = "0.08,0.08,0.08,0.08"
    random_perspective_transform = 0.04
    size_ = 32
    gaussian_prior_resizing = 3
    
    count_ = args.count

    params_list.append({"count": count_, "equal_char": True, "size": size_, "ensure_square_layout": True, 
                        "blur": 1, "image_mode": "RGB", "gaussian_prior_resizing": gaussian_prior_resizing,
                        "margins": margins, "background": "plain_white", 
                        "random_translation_x": True, "random_translation_y": True, 
                        "random_seed": 57, "image_blending_method": "trivial", 
                        "pre_elastic": pre_elastic, 
                        "rotation": "-30 30", "shear_x": "-0.2 0.2", 
                        "random_perspective_transform": random_perspective_transform})
    
    for idx, parameters in enumerate(params_list):
        generate_one_multilingual_dataset(args, parameters, idx)








