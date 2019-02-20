
# Beaver - Transformer based NMT

![](https://github.com/voidmagic/Beaver/blob/master/docs/biubiubiu.png)


Beaver is a simple and efficient neural machine translation system based on pytorch.
The design goal is to become a friend **start point** for NMT researchers.

## Features:
* **Simple**: the codebase is very concise(less than 1k lines of python code).
* **Complete**: beaver includes a **complete** pipeline from preprocess to translation, and most of the preprocess tools are borrowed from [moses](https://github.com/moses-smt/mosesdecoder) to guarantee standardization.
* **Efficient**: beaver is faster than most of current open source NMT systems. Multi GPU training and inference are already supported.
* **Modularized**: beaver includes 4 main modules(data, infer, loss and model) and a utility module(utils).


The complete document is coming soon...

## Before start
Clone this repository
```
git clone https://github.com/voidmagic/Beaver.git
```
Looking through the project
```
Beaver/
├── LICENSE
├── README.md
├── docs
├── beaver
│   ├── __init__.py
│   ├── data
│   ├── infer
│   ├── loss
│   ├── model
│   └── utils
├── tools
│   └── build_vocab.py
├── train.py
└── translate.py
```

Install requirements(only for python 3):
```
pip install torch==0.4.1 torchvision==0.2.1 subword-nmt==0.3.5
```

Add beaver path to environment:
```
export BEAVER=/PATH/TO/Beaver
export PYTHONPATH=$BEAVER:$PYTHONPATH
```

The following part will take iwslt 14 de-en data as example to get throuth Beaver.

## Data preparation

1. Download data from [IWSLT 14 DE-EN](https://wit3.fbk.eu/archive/2014-01//texts/de/en/de-en.tgz)
2. unzip this data
    ```
    de-en
    ├── IWSLT14.TED.dev2010.de-en.de.xml
    ├── IWSLT14.TED.dev2010.de-en.en.xml
    ├── IWSLT14.TED.tst2010.de-en.de.xml
    ├── IWSLT14.TED.tst2010.de-en.en.xml
    ├── IWSLT14.TED.tst2011.de-en.de.xml
    ├── IWSLT14.TED.tst2011.de-en.en.xml
    ├── IWSLT14.TED.tst2012.de-en.de.xml
    ├── IWSLT14.TED.tst2012.de-en.en.xml
    ├── IWSLT14.TEDX.dev2012.de-en.de.xml
    ├── IWSLT14.TEDX.dev2012.de-en.en.xml
    ├── README
    ├── train.en
    ├── train.tags.de-en.de
    └── train.tags.de-en.en

    0 directories, 14 files
    ```
3. clean data

    You may need mosesdecoder to preprocess data:
    ```
    git clone --depth=1 https://github.com/moses-smt/mosesdecoder.git tools/mosesdecoder
    ```
    ```
    # process xml
    perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' <  de-en/IWSLT14.TED.dev2010.de-en.de.xml > valid.raw.src
    perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' <  de-en/IWSLT14.TED.dev2010.de-en.en.xml > valid.raw.tgt
    cat de-en/IWSLT14.TED.tst201*.de-en.de.xml | perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' > test.raw.src
    cat de-en/IWSLT14.TED.tst201*.de-en.en.xml | perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' > test.raw.tgt
    cp de-en/train.tags.de-en.de train.raw.src
    cp de-en/train.tags.de-en.en train.raw.tgt

    # normalize punctuation
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < test.raw.src >  test.norm.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < test.raw.tgt >  test.norm.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < valid.raw.src > valid.norm.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < valid.raw.tgt > valid.norm.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < train.raw.src > train.norm.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < train.raw.tgt > train.norm.tgt

    # remove non-printing char
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < test.norm.src  > test.np.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < test.norm.tgt  > test.np.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < valid.norm.src > valid.np.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < valid.norm.tgt > valid.np.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < train.norm.src > train.np.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < train.norm.tgt > train.np.tgt

    # lowercase
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < test.np.src > test.lc.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < test.np.tgt > test.lc.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < valid.np.src > valid.lc.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < valid.np.tgt > valid.lc.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < train.np.src > train.lc.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < train.np.tgt > train.lc.tgt

    # tokenize
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < test.lc.src > test.tok.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < test.lc.tgt > test.tok.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < valid.lc.src > valid.tok.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < valid.lc.tgt > valid.tok.tgt

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < train.lc.src > train.tok.src
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < train.lc.tgt > train.tok.tgt
    ```
4. apply bpe

    ```
    # learn bpe and apply to all data
    subword-nmt learn-joint-bpe-and-vocab --input train.tok.src train.tok.tgt -s 8000 -o codes.share --write-vocabulary vocab.src vocab.tgt
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < train.tok.src > train.bpe.src
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < train.tok.tgt > train.bpe.tgt
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < valid.tok.src > valid.bpe.src
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < valid.tok.tgt > valid.bpe.tgt
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < test.tok.src > test.bpe.src
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < test.tok.tgt > test.bpe.tgt
    ```

5. filter training corpus by length
    ```
    perl ${BEAVER}/tools/mosesdecoder/scripts/training/clean-corpus-n.perl train.bpe src tgt train.clean 1 100 -ratio 1.5
    perl ${BEAVER}/tools/mosesdecoder/scripts/training/clean-corpus-n.perl valid.bpe src tgt valid.clean 1 100 -ratio 1.5
    ```

6. build vocabulary

    Build vocabulary for each language:
    ```
    cat train.clean.src | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.src
    cat train.clean.tgt | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.tgt
    ```
    Or if you want a shared vocabulary:
    ```
    cat train.clean.src train.clean.tgt | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.share
    ```
## Training

    ```
    python3 ${BEAVER}/train.py -train train.clean.src train.clean.tgt -valid valid.clean.src valid.clean.tgt -vocab vocab.8k.share 
    ```

The full parameters can be found in `beaver/utils/parseopt.py`.
After training, the model will saved into ${MODEL_PATH}, which is `train-${TIMESTAPE}` by default.

## Translation
    ```
    python3 ${BEAVER}/translate.py -input test.bpe.src -vocab vocab.8k.share -model_path ${MODEL_PATH}
    ```

## TODOs:

1. More hyper-parameters should be exposed.
2. Comparision with existing open source projects. 

## References:
* [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
* [THUMT: An Open Source Toolkit for Neural Machine Translation](https://github.com/thumt/THUMT)
* [OpenNMT-py: Open-Source Neural Machine Translation](https://github.com/OpenNMT/OpenNMT-py)

## License
This project is under [The 3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).

