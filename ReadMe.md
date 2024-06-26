Codebase for "SymKGQA: Few-Shot Knowledge Graph Question Answering via Symbolic Program Generation and Execution"

## Citation
```
@inproceedings{
anonymous2024symkgqa,
title={Sym{KGQA}: Few-Shot Knowledge Graph Question Answering via Symbolic Program Generation and Execution},
author={Anonymous},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=nwlIPR4NwR}
}
```


# KoPL Generation and Execution

This repository contains tools for generating and executing KoPL programs across various datasets. Below are the details of the folder structure and instructions for running the scripts.

## Folder Structure

1. **KoPL Generation**: This folder contains scripts for generating KoPL programs in both manual and dynamic settings.
    - **Manual Demonstrations**: Scripts for generating KoPL programs manually.
    - **Dynamic Demonstrations**: Scripts for generating KoPL programs dynamically, with subdirectories for different datasets (e.g., KQA Pro).

2. **QUACK**: This folder contains the executor for querying KoPL programs over a knowledge base (KB) to retrieve answers.

3. **data**: This folder contains input data files for three datasets: KQA Pro, MetaQA, and WebQSP.

## How to Use

### Generating KoPL Programs

#### KQA Pro Dataset

**Manual Setting**:
1. Navigate to the `KoPL Generation/Manual Demonstrations` directory.
2. Run the following command:
   ```sh
   cd KoPL Generation/Manual Demonstrations
   python3 palm_kqapro.py
   ```

**Dynamic Setting**:
1. Navigate to the `KoPL Generation/Dynamic Demonstrations/kqapro` directory.
2. Run the following command:
   ```sh
   cd KoPL Generation/Dynamic Demonstrations/kqapro
   python3 palm2_dynamic.py
   ```

#### Other Datasets (webQSP and MetaQA)

You can generate KoPL programs for the webQSP and MetaQA datasets similarly by navigating to their respective directories and running the appropriate scripts.

### Executing KoPL Programs

#### KQA Pro Dataset

1. Navigate to the `QUACK` directory.
2. Run the following command to execute KoPL programs:
   ```sh
   cd QUACK
   python -m Program.executor_rule_KQAPro 'input_file_path' 'ground_file_path' > 'log_file_path'
   ```

#### Other Datasets (webQSP and MetaQA)

You can execute programs generated for the webQSP and MetaQA datasets similarly by navigating to their respective directories and running the appropriate scripts.

## Data

The `data` folder contains input data files for the following datasets:
- KQA Pro
- MetaQA
- WebQSP

Ensure that the input files are correctly placed in the `data` folder before running the scripts.

---

This README provides an overview and basic usage instructions for generating and executing KoPL programs. For more detailed information.
