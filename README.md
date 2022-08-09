pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd Models
git clone https://github.com/siat-nlp/GALAXY.git
<!-- wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=18NPZQ6SH9Q0nFZenf_hNyuJTyT9IFAjL' -O 'model.zip'
unzip model.zip -->
<!-- ### Data Preparation
Download data from this [link](https://drive.google.com/file/d/1oi1w_zNH-GAMfav6slVXIF1usALHJJNQ/view?usp=sharing). 

The downloaded zip file `data.zip` contains pre-training corpora and four TOD benchmark datasets: MultiWOZ2.0, MultiWOZ2.1, In-Car Assistant and CamRest, which have already been processed. You need to put the unzipped directory `data` into the project directory `GALAXY` for the subsequent training. -->


# GALAXY
## Pre-training

### Pre-trained Checkpoint
- [GALAXY](https://drive.google.com/file/d/18NPZQ6SH9Q0nFZenf_hNyuJTyT9IFAjL/view?usp=sharing): an uncased model with DA classification head (12-layers, 768-hidden, 12-heads, 109M parameters)

You need to unzip the downloaded model file `model.zip`, then put the unzipped directory `model` into the project directory `GALAXY` for the futhuer fine-tuning.

### Fine-tuned Checkpoints
Download checkpoints from this [link](https://drive.google.com/file/d/1JerSwvLzes6b-igQ7lPCTIrh6IvrTMK6/view?usp=sharing). 

The downloaded zip file `outputs.zip` contains the best fine-tuned checkpoints on different datasets: 
- the **7-th** epoch on MultiWOZ2.0 (**60** training epochs in total)
- the **5-th** epoch on MultiWOZ2.1 (**60** training epochs in total)

