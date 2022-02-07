The idea of the project is to create composite templates which will ultimately extend the coverage and at the same time make the model more real and efficient. 
The natural language question templates created are passed through the SubjectiveEye3D model to check the viability of the question i.e whether they make sense or not. 
Finally, a NMT model is used to translate the NLQ to it's respective sparql query. I tried to implement them steps in order to gain a better understanding of the 
project by reading the bog post by Anand Panchbhai- https://anandpanchbhai.com/A-Neural-QA-Model-for-DBpedia/ and the entire code can be found here- 
https://github.com/dbpedia/neural-qa/tree/gsoc-anand

1) **Generate URL:** An OWL file containing all the ontology related details was used to generate a URL for a specific label. For instance, for label Monument, it's URL is http://mappings.dbpedia.org/server/ontology/classes/Monument 
2) **Fetch Property Data:** The Monument ontology class http://mappings.dbpedia.org/server/ontology/classes/Monument is considered and all its properties are fetched for experimentation purposes.
3) **Sentence and Template Generator:** This is an integral step because not only the simple templates but also composite templates were generated. Simple template- What is the  alma mater of <A> ? and it's sparql query- select ?x where { <A>  dbo:almaMater ?x  } .Composite template- Who is the  alumni of alma mater of <A> ? and it's sparql query- select ?x where { <A>  dbo:almaMater ?x1 . ?x1 dbo:alumni ?x  } 
4) **Fetch Ranks:** Some of the NLQ templates did not make any sense and hence needed to be filtered. The NLQ templates created are passed through the SubjectiveEye3D model to check the viability of the question i.e whether they make sense or not. For it to be filtered a threshold score was required."
5) **Generator Functionality:** The templates prepared in the previous step were used to fetch examples using sparql endpoint. Using this step, the data.en and the data.sparql file were created. For instance, an example from data.en is What is the architect of tour de la bourse ?, and of data.sparql is select var_x where brack_open dbr_Tour_de_la_Bourse dbo_architect var_x brack_close
6) **Train_dev_test Split:** The data.en and data.sparql were split into 3 files respectively.
7) **Train:** The nspm project directory was considered to train the model. Due to GPU memory constraints, I was only able to train a model with a batch size of 4. But such a small batch size is not good as it does not represent the entire dataset. Hence, I implemented Gradient Accumulation which virtually allows me to consider batch size of 32 atleast(not tried above 32)
8) **Interpret the test data:** Predictions are made on the test data specifically the test.en file which contains the NLQ examples.
9) **Scoring:** bleu score of the results is generated.

Experimentation
=================
Monument dataset is used for experimentation purposes. Above steps are followed from start to finish. Also number of examples per template are maximum 300. Below are the results from the experiment:
 
**Loss:**
Iterations  | Keras Tokenizer
------------- | -------------
750 | 0.8372
1500 | 0.4646
2250 | 0.3705
3000 | 0.3236
3750 | 0.2889

**Score:**
 Score Type | Keras Tokenizer
------------- | -------------
bleu score | 39.40

