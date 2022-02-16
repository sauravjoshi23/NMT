The idea of the project is to add a Paraphraser module which will ultimately extend the coverage and at the same time make the model more real and efficient. 
I tried to implement them steps in order to gain a better understanding of the project by reading the bog post by Zheyuan BAI- 
https://baiblanc.github.io/2020/08/27/GSOC-Final-Report/ and the entire code can be found here- https://github.com/dbpedia/neural-qa/tree/gsoc-zheyuan

1) **Generate URL:** An OWL file containing all the ontology related details was used to generate a URL for a specific label. For instance, for label Monument, it's URL is http://mappings.dbpedia.org/server/ontology/classes/Monument 
2) **Fetch Property Data:** The Monument ontology class http://mappings.dbpedia.org/server/ontology/classes/Monument is considered and all its properties are fetched for experimentation purposes.
3) **Sentence and Template Generator:** This is an integral step because not only the simple templates but also it paraphraser templates are generated. Firstly, paraphraser candidates are generated using the Text-To-Text Transfer Transformer model, then 2 techniques are used to select the candidates. First, using the a similarity metric and second, using a bert text classification method to classify the paraphrases i.e human-like or not. 
4) **Fetch Ranks:** Some of the NLQ templates did not make any sense and hence needed to be filtered. The NLQ templates created are passed through the SubjectiveEye3D model to check the viability of the question i.e whether they make sense or not. For it to be filtered a threshold score was required."
5) **Generator Functionality:** The templates prepared in the previous step were used to fetch examples using sparql endpoint. Using this step, the data.en and the data.sparql file were created. For instance, an example from data.en is What is the architect of tour de la bourse ?, and of data.sparql is select var_x where brack_open dbr_Tour_de_la_Bourse dbo_architect var_x brack_close
6) **Train_dev_test Split:** The data.en and data.sparql were split into 3 files respectively.
7) **GloVe Embeddings:** The data.en is fine-tuned with the existing GloVe embeddings whereas for data.sparql, GloVe embeddings are made from scratch.
8) **Train:** The nspm project directory was considered to train the model. Due to GPU memory constraints, I was only able to train a model with a batch size of 4. But such a small batch size is not good as it does not represent the entire dataset. Hence, I implemented Gradient Accumulation which virtually allows me to consider batch size of 32 atleast(not tried above 32)
9) **Interpret the test data:** Predictions are made on the test data specifically the test_benchmark.en file which contains the NLQ examples.
10) **Scoring:** bleu score of the results is generated.

Experimentation
=================
Monument dataset is used for experimentation purposes. Above steps are followed from start to finish. Also number of examples per template are maximum 300. Below are the results from the experiment:
 
**No Paraphraser Loss:**
Iterations | Keras Tokenizer | GloVe
------------- | ------------- | -------------
250	| 1.3027 | 1.24
500	| 0.7486 | 0.73
750	| 0.7225 | 0.64
1000 | 0.6107 | 0.59
1250 | 0.5716 | 0.51
1500 | 0.5412 | 0.47
1750 | 0.5151 | 0.44
2000 | 0.4994 | 0.41
2250 | 0.4774 | 0.37
2500 | 0.465 | 0.34

**Paraphraser Loss:**
Iterations  | Keras Tokenizer | GloVe
------------- | ------------- | -------------
700 | 1.1024 | 1.053
1400 | 0.7705 | 0.6287
2100 | 0.6622 | 0.490
2800 | 0.5785 | 0.421

**Score:**
 Score Type | Embeddings | Paraphraser | Score
------------- | ------------- | ------------- | -------------
bleu score | Keras Tokenizer | No | 41.14
bleu score | Keras Tokenizer | Yes | 41.36
bleu score | GloVe | No | 41.57
bleu score | GloVe | Yes | 42.65

The bleu score for the last rows specifically are very close because the dataset was too small and the paraphrases were mostly similar to original candidates as the bert classification dataset was trained on a small dataset. 

