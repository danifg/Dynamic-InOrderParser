# Dynamic In-Order Parser
This repository includes the code of the in-order parser trained with a dynamic oracle described in EMNLP paper [Dynamic Oracles for Top-Down and In-Order Shift-Reduce Constituent Parsing](https://aclanthology.info/papers/D18-1161/d18-1161). The implementation is based on the in-order parser (https://github.com/LeonCrashCode/InOrderParser) and reuses part of its code, including data preparation and evaluating scripts.

This implementation requires the [cnn library](https://github.com/clab/cnn-v1) and you can find pretrained word embeddings for English and Chinese in https://github.com/LeonCrashCode/InOrderParser. 

## Building
The boost version is 1.5.4.

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make

## Experiments

#### Data

You could use the scripts to convert the format of training, development and test data, respectively.

    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [training data in bracketed format] > [training oracle]
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [development data in bracketed format] > [development oracle]   
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [test data in bracketed format] > [test oracle]

#### Training

    mkdir model/
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --dev_data [development oracle] --bracketing_dev_data [development data in bracketed format] -P -t --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -D 0.2

#### Test
    
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --test_data [test oracle] --bracketing_dev_data [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file]

The automatically generated file test.eval is the result file.

For more information, please visit https://github.com/LeonCrashCode/InOrderParser.

## Citation
    @inproceedings{fernandez-gonzalez-gomez-rodriguez-2018-dynamic,
        title = "Dynamic Oracles for Top-Down and In-Order Shift-Reduce Constituent Parsing",
        author = "Fern{\'a}ndez-Gonz{\'a}lez, Daniel  and G{\'o}mez-Rodr{\'\i}guez, Carlos",
        booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
        month = oct # "-" # nov,
        year = "2018",
        address = "Brussels, Belgium",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D18-1161",
        doi = "10.18653/v1/D18-1161",
        pages = "1303--1313"    
    }
    
## Acknowledgments

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150), from MINECO (FFI2014-51978-C2-2-R, TIN2017-85160-C2-1-R) and from Xunta de Galicia (ED431B 2017/01).
