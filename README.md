# OmniPrint-NeurIPS-paper-experiments

This repository contains the code to reproduce the experiments described in the paper [OmniPrint: A Configurable Printed Character Synthesizer](https://openreview.net/forum?id=R07XwJPmgpl), which is accepted at NeurIPS 2021 Track Datasets and Benchmarks. 

The main repository of OmniPrint is [https://github.com/SunHaozhe/OmniPrint](https://github.com/SunHaozhe/OmniPrint)


## Instructions

```bash
git clone https://github.com/SunHaozhe/OmniPrint
git clone https://github.com/SunHaozhe/OmniPrint-NeurIPS-paper-experiments
```

Copy and paste each subfolder of `OmniPrint-NeurIPS-paper-experiments/experiments/` into `OmniPrint/`. 


* `OmniPrint-NeurIPS-paper-experiments/experiments/baseline_algorithms`: Section 4.1 Few-shot learning
* `OmniPrint-NeurIPS-paper-experiments/experiments/baseline_algorithms_vary_train_size`: Section 4.3 Influence of the number of meta-training episodes for few-shot learning
* `OmniPrint-NeurIPS-paper-experiments/experiments/baseline_algorithms_Z`: Section 4.2 Other meta-learning paradigms
* `OmniPrint-NeurIPS-paper-experiments/experiments/regression`: Section 4.5 Character image regression tasks
* `OmniPrint-NeurIPS-paper-experiments/experiments/transfer`: Section 4.4 Domain adaptation
* `OmniPrint-NeurIPS-paper-experiments/experiments/transfer_mnist`: Section 4.4 Domain adaptation


For the slurm scripts, some `sbatch` parameters have been removed (cluster partition, cluster QoS, which nodes to use). Please complete and adapt them according to your compute resources. 


## Citation

```
@inproceedings{sun2021omniprint,
title={OmniPrint: A Configurable Printed Character Synthesizer},
author={Haozhe Sun and Wei-Wei Tu and Isabelle M Guyon},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
year={2021},
url={https://openreview.net/forum?id=R07XwJPmgpl}
}
```

