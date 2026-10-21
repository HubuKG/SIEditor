# SIEditor

SIEditor is a knowledge editing method for pre-trained language model based
knowledge graph embeddings (PLM-based KGEs), originally developed within the
[DeltaKG](https://github.com/zjunlp/PromptKG/tree/main/deltaKG) library.

This repository contains only the code needed to train and evaluate SIEditor.

## Repository structure

```
.
├── deltaKG/
│   ├── models/KGEditor/run.py      # training entry point (shared runner)
│   ├── src/
│   │   ├── models/                 # SIEditor / KGEditor model implementations
│   │   └── data/                   # dataset & preprocessing code
│   ├── scripts/SIEditor/           # launch scripts (main / ablation / sequential)
│   ├── checkpoints/                # downloaded pretrained KGE checkpoints (not tracked)
│   ├── datasets/                   # downloaded task data (not tracked)
│   ├── resource/model.png
│   ├── requirements.txt
│   ├── run.sh
│   └── README.md                   # original DeltaKG readme
└── LICENSE
```

> `models/KGEditor/run.py` is the shared training entry point.
> SIEditor is selected via `--kge_model_type SIEditor`.

## Installation

```bash
conda create -n deltakg python=3.8
conda activate deltakg
cd deltaKG
pip install -r requirements.txt
```

`requirements.txt` does not pin PyTorch. Install a version compatible with your
CUDA driver first, e.g. `torch==1.12.0+cu113`.

## Data & checkpoints

The datasets and pretrained KGE checkpoints are large and are **not** included
in this repository. Download them from the links below and place them under
`deltaKG/datasets/` and `deltaKG/checkpoints/`.

- Datasets: [Google Drive](https://drive.google.com/drive/folders/1K1gag6eTCJ-x7WM-zbn6IRGX_B9fy4hD?usp=share_link)
- Checkpoints: [Google Drive](https://drive.google.com/drive/folders/1EOHdg8rC9iwgSyKl5RnEv9z6ATW5Ntbr?usp=share_link)

Expected layout:

```
deltaKG/
├── checkpoints/
│   ├── PT_KGE_A-FB15k237/
│   ├── PT_KGE_E-FB15k237/
│   ├── PT_KGE_A-WN18RR/
│   └── PT_KGE_E-WN18RR/
└── datasets/
    ├── FB15k237/
    │   ├── AddKnowledge/     # train.jsonl, dev.jsonl, test.jsonl, stable.jsonl,
    │   │                     # relations.txt, relation2text.txt, entity2textlong.txt
    │   └── EditKnowledge/
    └── WN18RR/
        ├── AddKnowledge/
        └── EditKnowledge/
```

## Running SIEditor

```bash
cd deltaKG

# Directly with a launch script
bash scripts/SIEditor/SIEditor_FB15k237_add.sh
bash scripts/SIEditor/SIEditor_FB15k237_edit.sh
bash scripts/SIEditor/SIEditor_WN18RR_add.sh
bash scripts/SIEditor/SIEditor_WN18RR_edit.sh

# Or via the quick-start script
bash run.sh -m SIEditor -d FB15k237 -t add
```

Ablation and sequential-editing experiments are under
`deltaKG/scripts/SIEditor/ablation/` and `deltaKG/scripts/SIEditor/sequential/`.

Training logs and checkpoints are written to the `--dirpath` specified in each
script (e.g. `models/SIEditor/logger/A-FB15k237`).

## License

MIT License. See [LICENSE](LICENSE).