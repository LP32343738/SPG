# Style-Preserving Generator to Make Synthetic License Plate Data
![SPG.jpg](SPG.jpg)
> **Abstract:** We propose the Style-Preserving Generator (SPG) to make synthetic license plate data and address the data imbalance and privacy issues when making a License Plate Recognition (LPR) model. The proposed SPG can edit the characters on real-world license plates while maintaining their original styles, enabling the generation of synthetic license plate data with user-specified characters. We can thus synthesize license plates with desired characters to effectively alleviate the data imbalance and privacy issues associated with real-world license plates. The SPG consists of several components: a transformer, a source encoder, a source style encoder, a character mask decoder, a target generator, and a target discriminator. Given a source license plate image and a specified text as inputs, these components are working together to compute the self- and cross-attention embeddings, predict character masks, and generate a synthetic license plate in the source style but with source characters replaced by the specified characters. We adopt a two-phase training scheme. Phase 1 involves using synthetic training data with ground-truth character masks available, followed by Phase 2 hybrid training using both synthetic and real-life data without the ground-truth character masks of the latter. To showcase the effectiveness of the SPG, we introduce a new benchmark dataset, namely the LP-2024 (License Plate 2024), which alleviates the limitations of existing datasets and presents new challenges for license plate recognition and generative models. We validate the SPG’s performance on the LP-2024 dataset and other benchmark datasets, and compare it against state-of-the-art text editing approaches.





# Getting Started
- Clone the repo:
```
git clone https://github.com/LP32343738/SPG.git
cd SPG
```
# Installation
- Python 3.7
- Pytorch 1.11.0
2. Install the requirements
```
pip install -r requirements.txt
```

# LP2024 Dataset
Please visit the [LP2024](https://github.com/LP32343738/LP2024) for detailed information and download instructions.

# CCPD Syn Dataset
If you wish to obtain the dataset, please send a hard drive to the laboratory of Prof. Gee-Sern (Jison) Hsu, Department of Mechanical Engineering, National Taiwan University of Science and Technology.

# Pretrained Models
Get the pretrained models from GoogleDrive.
[GDrive](https://drive.google.com/file/d/1M2tJ1k5iHhwPt_o-y9JrQktWWH59mxp0/view?usp=sharing)

Please place the checkpoint files in the `./` directory.


# Training

phase1
```
python train_all.py 
```

phase2
```
python train_all_finetune.py 
```


# Inference
```
python predict.py --input_dir=i_s --input_text=gt.txt --checkpoint=./train_step-xxx.model
```
Additional flags:
- `--input_dir /path/to/i_s` set up an image for license plate text replacement
- `--input_text /path/to/gt.txt` set the text to be replaced
- `--checkpoint ` set the path of trained model

# Demo Pretrained Model
|Demo Pretrained Model|
|---|
|[GDrive](https://drive.google.com/file/d/118EZBG9g3EN1SX89MaWk_RvhXQeBcZxn/view?usp=sharing)|


    
