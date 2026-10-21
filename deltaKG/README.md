<div align="center">
    <img src="" width="550px">
    <p> 
        A Library for Dynamically Editing PLMs-Based Knowledge Graph Embeddings.
    </p>
    
------

</div>


## Overview
Knowledge graph embedding (KGE) is a method for representing symbolic facts in low-dimensional vector spaces, with the goal of projecting relations and entities into a continuous vector space. This approach enhances knowledge reasoning capabilities and facilitates application to downstream tasks.

We introduce DeltaKG (MIT License), a dynamic, PLM-based library for KGEs that equips with numerous baseline models, such as K-Adapter, CaliNet, KnowledgeEditor, MEND, KGEditor and SIEditor, and supports a variety of datasets, including E-FB15k237, A-FB15k237, E-WN18RR, and A-WN18RR.

**DeltaKG** is now publicly open-sourced, with [a demo](https://huggingface.co/spaces/zjunlp/KGEditor), [a leaderboard](https://zjunlp.github.io/project/KGE_Editing/) and long-term maintenance.
<!-- - ❗NOTE: We provide some KGE baselines at [OpenBG-IMG](https://github.com/OpenBGBenchmark/OpenBG-IMG). -->

## Installation

**Step1** Download the basic code

```bash
git clone --depth 1 
```

**Step2** Create a virtual environment using `Anaconda` and enter it
```bash
conda create -n deltakg python=3.8
conda activate deltakg
```
**Step3** Enter the task directory and install library
```bash
cd PromptKG/deltaKG
pip install -r requirements.txt
```

## Data & Checkpoints Download

### Data

The datasets that we used in our experiments are as follows,

- E-FB15k237

    This dataset is based on FB15k237 and a pre-trained language-model-based KGE. You can download the E-FB15k237 dataset from [Google Drive](https://drive.google.com/drive/folders/1K1gag6eTCJ-x7WM-zbn6IRGX_B9fy4hD?usp=share_link).

For other datasets `A-FB15k237`, `E-WN18RR`, and `A-WN18RR`, you can also download from the above link.

### Checkpoints

The checkpoints that we used in our experiments are as follows,

- PT_KGE_E-FB15k237

    This checkpoint is based on FB15k237 and a pre-trained language model. You can download the PT_KGE_E-FB15k237 checkpoint from [Google Drive](https://drive.google.com/drive/folders/1EOHdg8rC9iwgSyKl5RnEv9z6ATW5Ntbr?usp=share_link).

For other checkpoints `PT_KGE_A-FB15k237`, `PT_KGE_E-WN18RR`, and `PT_KGE_A-WN18RR`, you can also download from the above link.

The expected structure of files is:

```
DeltaKG
 |-- checkpoints  # checkpoints for tasks
 |-- datasets  # task data
 |    |-- FB15k237  # dataset's name
 |    |    |-- AddKnowledge  # data for add task, A-FB15k237
 |    |    |    |-- train.jsonl    
 |    |    |    |-- dev.jsonl     
 |    |    |    |-- test.jsonl     
 |    |    |    |-- stable.jsonl  
 |    |    |    |-- relation2text.txt  
 |    |    |    |-- relation.txt 
 |    |    |    |-- entity2textlong.txt   
 |    |    |-- EditKnowledge # data for edit task, E-FB15k237
 |    |    |    |-- ... #  consistent with A-FB15k237
 |    |-- WN18RR  # dataset's name
 |    |    |-- AddKnowledge  # data for add task, A-WN18RR
 |    |    |    |-- train.jsonl    
 |    |    |    |-- dev.jsonl     
 |    |    |    |-- test.jsonl     
 |    |    |    |-- stable.jsonl  
 |    |    |    |-- relation2text.txt  
 |    |    |    |-- relation.txt 
 |    |    |    |-- entity2text.txt   
 |    |    |-- EditKnowledge # data for edit task, E-WN18RR
 |    |    |    |-- ... #  consistent with A-WN18RR
 |-- models  # KGEditor and baselines
 |    |-- CaliNet
 |    |    |-- run.py    
 |    |    |-- ...
 |    |-- K-Adapter
 |    |-- KE  # KnowledgeEditor
 |    |-- KGEditor
 |    |-- MEND
 |-- resource  # image resource
 |-- scripts  # running scripts
 |    |-- CaliNet
 |    |    |-- CaliNet_FB15k237_edit.sh
 |    |    |-- ...
 |    |-- K-Adapter
 |    |-- KE  # KnowledgeEditor
 |    |-- KGEditor
 |    |-- MEND
 |-- src
 |    |-- data       # data process functions
 |    |-- models     # source code of models
 |-- README.md
 |-- requirements.txt
 |-- run.sh  #  script to quick start

```

## How to run

- ### script

  - The script `run.sh` has three arguments `-m`, `-d`, and `-t`, which stand for model, dataset, and task.
    - `-m`: should be the name of a model in models (e.g. `SIEditor`, `MEND`, `KE`);
    - `-d`: should be either `FB15k237` or `WN18RR`;
    - `-t`: should be either `edit`.

- ### Edit Task

  - To train the `SIEditor` model in the paper on the dataset `E-FB15k237`, run the command below.

    ```shell
        bash run.sh -m SIEditor -d FB15k237 -t edit
    ```
  
  - To train the `SIEditor` model in the paper on the dataset `E-WN18RR`, run the command below.

    ```shell
        bash run.sh -m SIEditor -d WN18RR -t edit
    ```


## Other KG Representation Open-Source Projects

- [OpenKE](https://github.com/thunlp/OpenKE)
- [LibKGE](https://github.com/uma-pi1/kge)
- [CogKGE](https://github.com/jinzhuoran/CogKGE)
- [PyKEEN](https://github.com/pykeen/pykeen)
- [GraphVite](https://graphvite.io/)
- [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [CogDL](https://github.com/THUDM/cogdl)
- [NeuralKG](https://github.com/zjukg/NeuralKG)
- [KGxBoard](https://github.com/neulab/KGxBoard)
- [KGEditor](https://github.com/zjunlp/PromptKG/tree/main/deltaKG)
