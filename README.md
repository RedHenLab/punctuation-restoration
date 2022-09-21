# Punctuation Restoration using Transformer Models

This repository is built on the official implementation of the paper [*Punctuation Restoration using Transformer Models for High-and Low-Resource Languages*](https://aclanthology.org/2020.wnut-1.18/) accepted at the EMNLP workshop [W-NUT 2020](http://noisy-text.github.io/2020/).
The scripts contained here are adapted for use in the World Futures project. We made slight adaptions to the scripts in order to handle additional puncuation marks and to restore punctuation for Russian in the future.

## Data

#### English
English datasets are provided in `data/en` directory. The data used for the original tool developers were collected from [here](https://drive.google.com/file/d/0B13Cc1a7ebTuMElFWGlYcUlVZ0k/view).
For World Futures, we fine-tuned the tool to handle additional punctuation marks and thus had to retrain on custom data, for which we used data from the Brown Corpus Family. The train/test/dev files are in data/en/(dev|test|train)_brown_all_features_large.txt ("all features" refers to the extended feature set of punctuation marks; "large" refers to the fact that we trained on the entire Brown family as opposed to a subset). The other files are from the original repository and were left unchanged.

#### Scripts
the folder preprocessing_train contains the scripts that were used to prepare the training data, with two versions depending on which features are considered. The scripts assume that the training data is a .vrt file and output a file that can be used to train the tool.

#### Bangla
Bangla datasets are provided in `data/bn` directory and were left unchanged.

#### Russian
As of now, while we have added an option to reconstruct Russian punctuation, this has not been evaluated. The current test/train/dev data stems from a Russian newsscape corpus and, based on our experiments with English, will likely yield poor results.



## Model Architecture
We fine-tune a Transformer architecture based language model (e.g., BERT) for the punctuation restoration task.
Transformer encoder is followed by a bidirectional LSTM and linear layer that predicts target punctuation token at
each sequence position.
![](./assets/model_architectue.png)


## Dependencies
Install PyTorch following instructions from [PyTorch website](https://pytorch.org/get-started/locally/). Remaining
dependencies can be installed with the following command
```bash
pip install -r requirements.txt
```


## Training
To train punctuation restoration model with optimal parameter settings for English run the following command
```
python src/train.py --cuda=True --pretrained-model=roberta-large --freeze-bert=False --lstm-dim=-1 
--language=english --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out
```

The weights.pt file in /out contains the weights trained on the Brown family corpus with the larger set of punctuation marks using the pretrained RoBERTA large model.


To train for Bangla the corresponding command is
```
python src/train.py --cuda=True --pretrained-model=xlm-roberta-large --freeze-bert=False --lstm-dim=-1 
--language=bangla --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 
--alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out
```

#### Supported models for English
```
bert-base-uncased
bert-large-uncased
bert-base-multilingual-cased
bert-base-multilingual-uncased
xlm-mlm-en-2048
xlm-mlm-100-1280
roberta-base
roberta-large
distilbert-base-uncased
distilbert-base-multilingual-cased
xlm-roberta-base
xlm-roberta-large
albert-base-v1
albert-base-v2
albert-large-v2
```

#### Supported models for Bangla
```
bert-base-multilingual-cased
bert-base-multilingual-uncased
xlm-mlm-100-1280
distilbert-base-multilingual-cased
xlm-roberta-base
xlm-roberta-large
```

#### Supported models for Russian
```
bert-base-multilingual-cased
bert-base-multilingual-uncased
distilbert-base-multilingual-cased
xlm-mlm-100-1280
xlm-roberta-base
xlm-roberta-large
```

## Pretrained Models
You can find pretrained models for RoBERTa-large model with augmentation for English [here](https://drive.google.com/file/d/17BPcnHVhpQlsOTC8LEayIFFJ7WkL00cr/view?usp=sharing)  
XLM-RoBERTa-large model with augmentation for Bangla can be found [here](https://drive.google.com/file/d/1X2udyT1XYrmCNvWtFpT_6jrWsQejGCBW/view?usp=sharing)

## Inference
You can run inference on unprocessed text file to produce punctuated text using `inference` module. Note that if the 
text already contains punctuation marks, they are removed before inference. 

Example script for English:
```bash
python inference.py --pretrained-model=roberta-large --weight-path=roberta-large-en.pt --language=en 
--in-file=data/test_en.txt --out-file=data/test_en_out.txt
```
Using the original weights and inference script, this should create a text file with the following output:
```text
Tolkien drew on a wide array of influences including language, Christianity, mythology, including the Norse Völsunga saga, archaeology, especially at the Temple of Nodens, ancient and modern literature and personal experience. He was inspired primarily by his profession, philology. his work centred on the study of Old English literature, especially Beowulf, and he acknowledged its importance to his writings. 
```

Using our custom model weights and the inference script which adds sentence boundary tags and adds spaces before the inserted marks, we get the following output:

```
Tolkien drew on a wide array of influences , including language , Christianity , mythology , including the Norse Völsunga saga , archaeology , especially at the Temple of Nods , ancient and modern literature and personal experience .</s><s> He was inspired primarily by his profession philology .</s><s> his work centred on the study of Old English literature , especially Beowulf .</s><s> and he acknowledged its importance to his writings .</s><s> 

```

Similarly, For Bangla
```bash
python inference.py --pretrained-model=xlm-roberta-large --weight-path=xlm-roberta-large-bn.pt --language=bn  
--in-file=data/test_bn.txt --out-file=data/test_bn_out.txt
```
The expected output is
```text
বিংশ শতাব্দীর বাংলা মননে কাজী নজরুল ইসলামের মর্যাদা ও গুরুত্ব অপরিসীম। একাধারে কবি, সাহিত্যিক, সংগীতজ্ঞ, সাংবাদিক, সম্পাদক, রাজনীতিবিদ এবং সৈনিক হিসেবে অন্যায় ও অবিচারের বিরুদ্ধে নজরুল সর্বদাই ছিলেন সোচ্চার। তার কবিতা ও গানে এই মনোভাবই প্রতিফলিত হয়েছে। অগ্নিবীণা হাতে তার প্রবেশ, ধূমকেতুর মতো তার প্রকাশ। যেমন লেখাতে বিদ্রোহী, তেমনই জীবনে কাজেই "বিদ্রোহী কবি"। তার জন্ম ও মৃত্যুবার্ষিকী বিশেষ মর্যাদার সঙ্গে উভয় বাংলাতে প্রতি বৎসর উদযাপিত হয়ে থাকে। 
```

In the original setup, *Comma* includes commas, colons and dashes, *Period* includes full stops, exclamation marks 
and semicolons and *Question* is just question marks. 
For our adaptation, COMMA includes commas and semicolons. PERIOD is for periods and colons. QUESTION includes question marks and combinations of question and exclamation marks. EXCLAMATION captures exclamation marks, and DASH is for dashes.


## Test
Trained models can be tested on processed data using `test` module to prepare result.

For example, to test the best preforming English model run following command
```bash
python src/test.py --pretrained-model=roberta-large --lstm-dim=-1 --use-crf=False --data-path=data/test
--weight-path=weights/roberta-large-en.pt --sequence-length=256 --save-path=out
```
Please provide corresponding arguments for `pretrained-model`, `lstm-dim`, `use-crf` that were used during training the
model. This will run test for all data available in `data-path` directory.

## Adjustments

In order to add additional punctuation marks, edit punctuation_dict in scr/config.py and specify what should be inserted when that feature is predicted in the variable punctuation_map in src/inference.py.
In order to edit which datasets are used to train for a given language, edit src/train.py

## Citing the original work

```
@inproceedings{alam-etal-2020-punctuation,
    title = "Punctuation Restoration using Transformer Models for High-and Low-Resource Languages",
    author = "Alam, Tanvirul  and
      Khan, Akib  and
      Alam, Firoj",
    booktitle = "Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wnut-1.18",
    pages = "132--142",
}
```
