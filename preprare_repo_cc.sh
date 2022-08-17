cd Models
git clone https://github.com/siat-nlp/GALAXY.git
git clone https://github.com/awslabs/pptod.git

pip install nltk
python obtain_downloadables.py

mkdir bert_model
cd bert_model
git clone https://huggingface.co/bert-base-uncased