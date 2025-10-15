# VQGAN for Topology Optimization (TO) Problems
In Topology Optimization (TO) and related engineering applications, physics-constrained simulations are often used to optimize candidate designs given some set of boundary conditions. Yet such models are computationally expensive and do not guarantee convergence to a desired result, given the frequent non-convexity of the performance objective.

To address this, we propose an augmented **Vector-Quantized GAN (VQGAN)** that allows for effective compression of TO designs within a discrete latent space, known as a **codebook**, while preserving high reconstruction quality. Our code uses [dome272's implementation](https://github.com/dome272/VQGAN-pytorch) of VQGAN as a starting point and adds several expansions for working with TO data, running models on HPC, reproducible training and evaluation, latent space analysis, and more.

Our experiments use a new dataset of two-dimensional heat sink designs optimized via Multi-physics Topology Optimization (MTO): [IDEALLab/MTO-2D dataset on Hugging Face](https://huggingface.co/datasets/IDEALLab/MTO-2D). We assume that VQGAN is always made conditional subject to a vector of scalar boundary conditions, material properties, etc. For example, the MTO problem specifies a condition vector of shape (3,) for each design, containing the fluid inlet velocity, maximum fluid power dissipation, and maximum fluid volume fraction, respectively.

We can also leverage the VQGAN codebook to train a small GPT-2 model, generating thermally performant heat sink designs within a fraction of the time taken by conventional optimization approaches.

---

## Train and evaluate VQGAN locally:
### Setup
0. Clone this repository and move to the project root directory.

1. Create & activate a Python 3.10 virtual environment

   **Windows (PowerShell)**  
   
   ```
   py -3.10 -m venv vqgan_env
   ```
   
   ```
   .\vqgan_env\Scripts\Activate.ps1
   ```
   
   **Linux / macOS**
   
   ```
   python3.10 -m venv vqgan_env
   ```
   
   ```
   source ./vqgan_env/bin/activate
   ```

2. Install the required PyTorch variant
   
   **CUDA 12.1 (GPU)**
   
      ```
      pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
      ```

   **CPU-only**
   
      ```
      pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
      ```

3. Install remaining dependencies  
   
   ``` 
   pip install -r requirements.txt --upgrade --no-cache-dir
   ```


### Training (VQGAN)

1. Train Stage 1 VQGAN using the HF dataset  

      ```
      python training_vqgan.py --load_from_hf True --run_name vqgan_stage_1
      ```

   Saves/checkpoints are then written to `../saves/vqgan_stage_1/` by default.

2. Train a smaller VQGAN (C-VQGAN) on the conditions; note that you must set `is_c = True` for this:

      ```
      python training_vqgan.py --load_from_hf True --is_c True --run_name cvqgan
      ```

### Evaluation (VQGAN)

1. Evaluate a trained VQGAN run
   
   ```
   python eval_vqgan.py --model_name vqgan_stage_1
   ```
   
---

## Training (Transformer)

1. Train a small GPT-2–style Transformer on VQGAN code sequences with CVQGAN conditioning

   ```
   python training_transformer.py --load_from_hf True --model_name vqgan_stage_1 --c_model_name cvqgan --run_name vqgan_stage_2
   ```

   Saves/checkpoints are then written to `../saves/vqgan_stage_2/` by default.

### Evaluation (Transformer)

1. Evaluate a trained Transformer run

   ```
   python eval_transformer.py --t_name vqgan_stage_2
   ```
   
---

## Citation
```bibtex
@inproceedings{drake2024quantize,
  title={To Quantize or Not to Quantize: Effects on Generative Models for 2D Heat Sink Design},
  author={Drake, Arthur and Wang, Jun and Chen, Qiuyi and Nejat, Ardalan and Guest, James and Fuge, Mark},
  booktitle={International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
  volume={88360},
  pages={V03AT03A017},
  year={2024},
  organization={American Society of Mechanical Engineers}
}
