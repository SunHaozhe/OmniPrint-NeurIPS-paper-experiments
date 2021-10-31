# Transfer learning benchmarking

### Workflow

##### Data

Use the script `omniprint/dataset/torch_image_dataset_formatter.py` to transform OmniPrint-meta[X] datasets to `ImageFolder` format (https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder). 

The 31 classes used in the paper were randomly sampled from the pool of 1409 classes. They are available at `OmniPrint-metaX-31.txt`


##### Transfer learning algorithm

```bash
git clone https://github.com/SunHaozhe/transferlearning.git
cd transferlearning/code/DeepDA
pip install -r requirements.txt
```

##### Beginning experiments

* Copy the bash scripts in this folder to `transferlearning/code/DeepDA`
* Edit the scripts to adapt the paths to the datasets
* Run the scripts (`bash ...`)


### Troubleshooting

If you encounter `ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`:

* Set `num-workers` to `0`
* Use the flag `--ipc=host` when executing `docker run ...`

See https://github.com/ultralytics/yolov3/issues/283 

