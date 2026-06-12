<div align="center">

# Text-Driven 3D Hand Motion Generation from Sign Language Data

<a href="https://leorebensabath.github.io"><strong>Léore Bensabath</strong></a>
·
<a href="https://mathis.petrovich.fr"><strong>Mathis Petrovich</strong></a>
·
<a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>

[![arXiv](https://img.shields.io/badge/arXiv-HandMDM-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2508.15902)
[![Project Page](https://img.shields.io/badge/Project_Page-HandMDM-blue.svg?logo=globe)](https://imagine.enpc.fr/~leore.bensabath/HandMDM)

</div>

The code for the HandMDM model and for the TMR model is largely inspired from the [STMC](https://github.com/nv-tlabs/stmc/tree/main) and [TMR](https://github.com/Mathux/TMR) repositories.  

## Data and model

### Model checkpoints

Download the model checkpoints from the [models folder](https://drive.google.com/drive/u/1/folders/1GeCAgX-tPAC8J9loeEmDSjCwjMBTLCor) and place it inside the HandMDM folder.

To evaluate the models using the retrieval metrics described in the paper, you also need to download the checkpoint for the THMR retrieval model. You can find it in the [THMR models folder](https://drive.google.com/drive/u/1/folders/1PbbCO57j0wJMfwgvdnWrGwaJpkNQETQC). Place this model folder inside `HandMDM/TMR`.

### Data

The motion dataset is currently waiting for publication under the official BOBSL webpage, which requires signing the BBC BOBSL - Terms of Use agreement. 
In the meantime, you can sign and send the agreement form following instructions at https://www.bbc.co.uk/rd/projects/extol-dataset. Then send me and email at leore.bensabath@enpc.fr with the countersigned agreement, and I will provide instructions to download the data. 

The annotations are available in the [annotations folder](https://drive.google.com/drive/u/1/folders/1GEfr1UeiOnXZnr12PmUlbu_jAqass6Yz). Download the annotations folder and put it inside `HandMDM/datasets/`. 

The text embeddings, already provided in the annotations folder, have been computed with command:
```bash
python -m prepare.text_stats dataset=bobsl3dt
```

You also need text and motion statistics data. They are available at the following locations:

- Text stats: [here](https://drive.google.com/drive/u/1/folders/1pAkqK1QQX2JjxIelSVsmCvIFNpBcBzsC).
They have been generated with command:
```bash
python -m prepare.text_stats dataset=bobsl3dt
```

- Motion stats: [here](https://drive.google.com/drive/u/1/folders/1b0Q-Ljgjumdwr8g1zZEHecth0-yYRBvA).
They have been generated with command:
```bash
python -m prepare.motion_stats dataset=bobsl3dt
```


## HandMDM

### Installation :construction_worker:
Clone and set up the environment as follows:

```bash
git clone https://github.com/leorebensabath/HandMDM
cd HandMDM/
```
This code was tested with python 3.9.21, cuda 11.8 and pytorch "2.4.1+cu118".

Creation of the environnement:
```bash
# create a virtual environnement (also works with conda)
python -m venv ~/.venv/handmdm
# activate the virtual environnement
source ~/.venv/handmdm/bin/activate
# upgrade pip
python -m pip install --upgrade pip
# Install pytorch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
# Install missing packages
python -m pip install -r requirements.txt
```

You also need `ffmpeg` available if you want to generate motion videos.

### Inference

You can generate motion from a text description using a trained checkpoint. Example: 

```bash
python inference.py checkpoint=models/mdm_bobsl3dt_phonology_hms/checkpoints/last.ckpt \
  input_text="The right hand, shaped as a fist with the thumb extended upwards, is positioned so the thumb tip touches the chin. The palm faces sideways"
```

The model was trained on motions of **14 frames** (at 25 fps).

**Config fields:**
- `checkpoint` — path to the `.ckpt` file (see [Model checkpoints](#model-checkpoints) above)
- `input_text` — text description of the motion to generate
- `length` — output motion length in frames (default: 14)
- `guidance` — classifier-free guidance weight (default: 15)
- `output_path` — where to save the motion `.npy` file (defaults to `<checkpoint_dir>/inference/<text>_<length>.npy`)
- `render` — if `true`, also saves an `.mp4` visualization next to the motion file
- `device` — e.g. `cuda` or `cpu`

### Evaluation

You can evaluate a trained HandMDM model on the test set using the THMR retrieval model (see [Model checkpoints](#model-checkpoints)). This command reproduces the **last row, right column, and right sub-columns of Table 2** in the paper (retrieval metrics and FID on `test_unseen`):

```bash
python -u eval.py run_dir=models/mdm_bobsl3dt_phonology_hms tmr_run_dir=models/tmr_bobsl3dt \
  dataset=bobsl3dt_test_manually_cleaned_phon_ms split=test_unseen seed=1234 \
  render=False ckpt=last tmr_ckpt=last
```

Defaults for other settings (e.g. `guidance=15`, `batch_size=512`) are in `configs/eval.yaml`. Override them on the command line as needed.

**Where outputs are saved**

With `{run_dir}` as `models/mdm_bobsl3dt_phonology_hms`, `{dataset}` as `bobsl3dtTestManuallyCleanedPhonMs` (camel-cased from the dataset name), `{split}` as `test_unseen`, `{guidance}` as `15`, `{epoch}` the diffusion checkpoint epoch, `{seed}` as `1234`, and `{tmr}` the TMR run folder name (e.g. `tmrBobsl3dt_epoch{N}`).

**Metrics** (always saved when `save_metrics=True`, the default):

| Output | Path |
|---|---|
| Retrieval, FID, and smoothness scores | `{run_dir}/metrics/{dataset}/{split}/guidance{guidance}_epoch{epoch}/{tmr}/metrics_{seed}.yaml` |
| Cached generated motions (reused on re-run) | `{run_dir}/metrics/{dataset}/{split}/guidance{guidance}_epoch{epoch}/inference_{seed}.pickle` |

The YAML file contains m2m/m2t R@1–R@10, FID, `acc_score`, and `smooth_score`, plus a LaTeX-ready summary string.

**Visuals** (only when `render=True`; with `run_dir` under `models/`, as in the example):

| Output | Path |
|---|---|
| Generated motion videos | `{run_dir}/eval_generations/{dataset}/{split}/guidance{guidance}_epoch{epoch}_seed{seed}/renderings/{keyid}.mp4` |
| Generated meshes | `{run_dir}/eval_generations/{dataset}/{split}/guidance{guidance}_epoch{epoch}_seed{seed}/meshes/{keyid}.npy` |
| Per-sample eval pickles | `{run_dir}/eval_generations/{dataset}/{split}/guidance{guidance}_epoch{epoch}_seed{seed}/TMR_{tmr}/{keyid}.pickle` |
| Ground-truth videos | `rendering_ground_truth/{motion_test_gt_folder}/{keyid}.mp4` |
| Ground-truth meshes | `rendering_ground_truth/meshes/{motion_test_gt_folder}/{keyid}.npy` |

With `render=False` (as in the example above), only the metrics and cached inference pickle are written.

### Training

You can train a HandMDM model on BOBSL using the following command. Replace `<run_name>` with a name for your experiment (e.g. `mdm_bobsl3dt`):

```bash
python -u train.py --config-name=train_bobsl experiment=bobsl3dt train_split=train val_split=null \
  dataset=bobsl3dt run_name=<run_name>
```

**Before training**, you need:

- The BOBSL data and annotations (see [Data](#data) above), with motions in LMDB format under `datasets/motions/lmdb--bobsl3dt`
- Precomputed CLIP text embeddings under `datasets/annotations/bobsl3dt/text_embeddings/`
- Motion and text normalizer statistics at `motion_stats/bobsl3dt/` and `text_stats/bobsl3dt/clip/` (compute with `prepare/motion_stats.py` and `prepare/text_stats.py` if not already present)

**What the command does**

- `experiment=bobsl3dt` — loads the BOBSL training recipe (274-dim motions, `drop_cond=0.05`, motion stats path, etc.)
- `val_split=null` — disables validation (validation is skipped when train and val splits are the same) - there is no validation set released for the BOBSL3DT dataset.
- Other defaults (batch size, learning rate, checkpoints) are in `configs/train_bobsl.yaml` and `configs/trainer.yaml`

**Where outputs are saved**

Everything for a run lives under `outputs/<run_name>/`:

| Output | Path |
|---|---|
| Saved training config | `outputs/<run_name>/config.json` |
| Training log | `outputs/<run_name>/train.out` |
| Metrics (CSV) | `outputs/<run_name>/logs/` |
| Checkpoints (`last.ckpt`, periodic `latest-{epoch}.ckpt`) | `outputs/<run_name>/logs/checkpoints/` |

Use `outputs/<run_name>/logs/checkpoints/last.ckpt` for [inference](#inference) or [evaluation](#evaluation) (set `run_dir` or `checkpoint` accordingly).

**Resuming** — set `resume_dir=outputs/<run_name>` and `ckpt=last` (or a specific checkpoint path) to continue a run. The experiment config sets `resume_dir` to the same run folder automatically when using `experiment=bobsl3dt`.


### Bibtex
If you use our code in your research, kindly cite our work:
```bibtex
@inproceedings{bensabath2026handmdm,
  title={Text-Driven 3D Hand Motion Generation from Sign Language Data},
  author={Bensabath, Léore and Petrovich, Mathis and Varol, G{\"u}l},
  booktitle={CVPR},
  year={2026}
}
```