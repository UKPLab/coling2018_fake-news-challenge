

![2010-07-07_ukp_banner](https://user-images.githubusercontent.com/29311022/27184688-27629126-51e3-11e7-9a23-276628da2430.png)

![aiphes_logo - small](https://user-images.githubusercontent.com/29311022/27278631-2e19f99e-54e2-11e7-919c-f89ae0c90648.png)
![tud_weblogo](https://user-images.githubusercontent.com/29311022/27184769-65c6583a-51e3-11e7-90e0-12a4bdf292e2.png)


## Repository of the COLING 2018 paper: [A Retrospective Analysis of the Fake News Challenge Stance Detection Task](https://arxiv.org/pdf/1806.05180.pdf)
* BibTeX:
	
		@inproceedings{tubiblio105434,
           		month = {Juni},
            		year = {2018},
       			booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
           		title = {A Retrospective Analysis of the Fake News Challenge Stance-Detection Task},
          		author = {Andreas Hanselowski and Avinesh P.V.S. and Benjamin Schiller and Felix Caspelherr and Debanjan * Chaudhuri and Christian M. Meyer and Iryna Gurevych},
             		url = {http://tubiblio.ulb.tu-darmstadt.de/105434/}
			}
			
## Introduction

The repository was originaly developed as a part of the Fake News Challenge Stage 1 (FNC-1 http://www.fakenewschallenge.org/) by team Athene:
[Andreas Hanselowski](mailto:hanselowski@aiphes.tu-darmstadt.de), [Avinesh PVS](mailto:avinesh@aiphes.tu-darmstadt.de), [Benjamin Schiller](mailto:schiller.benny@googlemail.com) and [Felix Caspelherr](mailto:felixc@shared-files.de).
In the project, we worked in close collaboration with [Debanjan Chaudhuri](mailto:deba.kgec@gmail.com). 

Prof. Dr. Iryna Gurevych, AIPHES-Ubiquitous Knowledge Processing (UKP) Lab, TU-Darmstadt, Germany

It was further developed and enhanced by [Felix Caspelherr](mailto:felixc@shared-files.de) in scope of his master thesis.
The code was additionaly modified and extended for the submission to the "[27th International Conference on Computational Linguistics (COLING 2018)](https://coling2018.org/)":
"[A Retrospective Analysis of the Fake News Challenge Stance Detection Task](https://arxiv.org/pdf/1806.05180.pdf)"


## Requirements

* Software dependencies
	
		python >= 3.4 (tested with 3.4)
	
## Installation

1. Install required python packages.

		python3.4 -m pip install -r requirements.txt --upgrade
		python3.4 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl
    
    alternatively you can set up an anaconda environment based on "anaconda_env_FNC_challenge.yml" by executing following command in the fnc folder:
        
        conda env create -f anaconda_env_FNC_challenge.yml
        
    (Note: If you use a higher CUDA version, you might have to use a newer version of tensorflow.)
        
2. Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

		python3.4 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('cmudict');"
	      
3. Copy Word2Vec [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) in folder 

        [project_name]/data/embeddings/google_news/ 
        
    (folders have to be created)

4. Download [Paraphrase Database: Lexical XL Paraphrases 1.0](http://www.cis.upenn.edu/~ccb/ppdb/release-1.0/ppdb-1.0-xl-lexical.gz) and extract it to the ppdb folder.
	
		gunzip ppdb-1.0-xl-lexical.gz [project_name]/data/ppdb/
		
    (folders have to be created)
        
5. To use the Stanford-parser an instance has to be started in parallel: Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html), extract anywhere and execute following command: 

		wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
		java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020
		
6. In order to reproduce the the results of the experiments mentioned in our COLING paper "A Retrospective Analysis of the Fake News Challenge Stance Detection Task", please modify fnc/settings.py to match the desired experiments.
At the current state, the model "voting_mlps_hard" will be trained and tested on the FNC corpus. All combinations of models and features used are listed in the settings file in _feature_list_.

## Additional notes

* Setup tested on Anaconda3 (tensorflow 0.9 gpu version)*
Be sure that cuda library is setup correctly.

		conda create -n env_python3.4 python=3.4 anaconda
		source activate env_python3.4
		python3.4 -m pip install -r requirements.txt --upgrade
		python3.4 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

	
## To Run

To run execute following steps:
1. Make sure that an instance of th Stanford CoreNLP server is running (see step 5 above)
2. run "python pipeline.py -p ftrain"
    <br /> --> features will be created and saved in the corresponding folder fnc/data/fnc-1/features/[selected corpus]/
    <br /> --> the model will be saved in /fnc/data/fnc-1/mlp_models/
3. note down the model name of the trained model and align settings.py
4. run "python pipeline.py -p ftest" to obtain results (FNC score, F1 scores and confusion matrix)
    <br /> --> Result-scores will appended to fnc/fnc_results.txt
    <br /> --> The labeled test-file will be saved to fnc/data/fnc-1/fnc_results/
 
Note: The "stanford features" and topic features may take several hours to be computed.

For more details
		
	python pipeline.py --help         
        
        e.g.: python pipeline.py -p crossv holdout ftrain ftest
        
        * crossv: runs 10-fold cross validation on train / validation set and prints the results
        * holdout: trains classifier on train and validation set, tests it on holdout set and prints the results
        * ftrain: trains classifier on train/validation/holdout set and saves it to fnc/data/fnc-1/mlp_models/
        * ftest: predicts stances of unlabeled test set based on the model

## System description

A more detailed description of the system including the features, which have been used, can be found in the document: [system_description_athene.pdf](https://github.com/hanselowski/athene_system/edit/master/system_description_athene.pdf)
