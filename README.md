
# Beaver - Transformer based NMT

![](https://github.com/voidmagic/Beaver/blob/master/docs/biubiubiu.png)


Beaver is a simple and efficient neural machine translation system based on pytorch.
The design goal is to become a friend **start point** for NMT researchers.

## Features:
* **Simple**: the codebase is very concise(less than 1k lines of python code).
* **Complete**: beaver includes a **complete** pipeline from preprocess to translation, and most of the preprocess tools are borrowed from [moses](https://github.com/moses-smt/mosesdecoder) to guarantee standardization.
* **Efficient**: beaver is faster than most of current open source NMT systems. The full comparision is coming soon and we welcome contributions.
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
    perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' <  de-en/IWSLT14.TED.dev2010.de-en.de.xml > valid.src
    perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' <  de-en/IWSLT14.TED.dev2010.de-en.en.xml > valid.tgt
    cat de-en/IWSLT14.TED.tst201*.de-en.de.xml | perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' > test.src
    cat de-en/IWSLT14.TED.tst201*.de-en.en.xml | perl -ne 'print $1."\n" if /<seg[^>]+>\s*(.*\S)\s*<.seg>/i;' > test.tgt
    cp de-en/train.tags.de-en.de train.src
    cp de-en/train.tags.de-en.en train.tgt

    # normalize punctuation
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < test.src >  test.src.norm
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < test.tgt >  test.tgt.norm

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < valid.src > valid.src.norm
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < valid.tgt > valid.tgt.norm

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de < train.src > train.src.norm
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < train.tgt > train.tgt.norm

    # remove non-printing char
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < test.src.norm  > test.src.np
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < test.tgt.norm  > test.tgt.np

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < valid.src.norm > valid.src.np
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < valid.tgt.norm > valid.tgt.np

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < train.src.norm > train.src.np
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl < train.tgt.norm > train.tgt.np

    # lowercase
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < test.src.np > test.src.lc
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < test.tgt.np > test.tgt.lc

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < valid.src.np > valid.src.lc
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < valid.tgt.np > valid.tgt.lc

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < train.src.np > train.src.lc
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/lowercase.perl < train.tgt.np > train.tgt.lc

    # tokenize
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < test.src.lc > test.src.tok
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < test.tgt.lc > test.tgt.tok

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < valid.src.lc > valid.src.tok
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < valid.tgt.lc > valid.tgt.tok

    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -thread 8 < train.src.lc > train.src.tok
    perl ${BEAVER}/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -thread 8 < train.tgt.lc > train.tgt.tok

    # remove unnecessary files
    rm *.norm *.lc *.np
    ```
4. apply bpe

    ```
    # learn bpe and apply to all data
    subword-nmt learn-joint-bpe-and-vocab --input train.src.tok train.tgt.tok -s 8000 -o codes.share --write-vocabulary vocab.src vocab.tgt
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < train.src.tok > train.src.bpe
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < train.tgt.tok > train.tgt.bpe
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < valid.src.tok > valid.src.bpe
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < valid.tgt.tok > valid.tgt.bpe
    subword-nmt apply-bpe --vocabulary vocab.src --vocabulary-threshold 50 -c codes.share < test.src.tok > test.src.bpe
    subword-nmt apply-bpe --vocabulary vocab.tgt --vocabulary-threshold 50 -c codes.share < test.tgt.tok > test.tgt.bpe
    ```

5. build vocabuary

    ```
    cat train.src.bpe | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.tgt
    cat train.tgt.bpe | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.src
    ```
    Or if you want a shared vocabuary:
    ```
    cat train.src.bpe train.tgt.bpe | python ${BEAVER}/tools/build_vocab.py 8000 > vocab.8k.share
    ```
## Training

    ```
    python3 ${BEAVER}/train.py -train train.src.bpe train.tgt.bpe -valid valid.src.bpe valid.tgt.bpe -vocab vocab.8k.share 
    ```

The full parameters can be found in `beaver/utils/parseopt.py`

## Translation
    ```
    python3 ${BEAVER}/translate.py -trans test.src.bpe test.tgt.bpe -vocab vocab.8k.share -model_path train
    ```

## TODOs:

1. More hyper-parameters should be exposed.
2. Implement Multi-GPU beam search(note that its already supported for training).
3. Model saving and loading logic.

## References:
* [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
* [THUMT: An Open Source Toolkit for Neural Machine Translation](https://github.com/thumt/THUMT)
* [OpenNMT-py: Open-Source Neural Machine Translation](https://github.com/OpenNMT/OpenNMT-py)

## License
This project is under [The 3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).

