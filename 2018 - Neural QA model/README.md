The idea of the project is to extend NSpM to cover more DBpedia classes to enable high quality question answering. It all starts with creating templates which is a combination of 2 different type of data - NLQ and Sparql. Multiple steps were involved in this project. I tried to implement them steps in order to gain a better understanding of the project by reading the bog post by Aman Mehta- https://amanmehta-maniac.github.io/ and the entire code can be found here- https://github.com/dbpedia/neural-qa/tree/gsoc-aman

1) **Fetch Property Data:** The Place ontology class http://mappings.dbpedia.org/server/ontology/classes/Place is considered and all its properties are fetched for experimentation purposes.
2) **Get number of occurences and URI:** It is essenstial to fetch the number of occurences of them properties along with their URI's because some properties might be noisy and should not be added into the dataset.
3) **Mapping step 1 and step 2**: The properties from step 1 and number of occurences and URI's are mapped to form a single csv file.
4) **Generate Minimal Viable Expression(MVE) for each property:** MVE is a question format generation of a property. For instance, for property "address", it's MVE is "What is the address of \<A>"
5) **Generate sparql query template and generator query template for each property:** SQT and GQT were prepared. For instance, for property "address", its SQT is "SELECT ?x WHERE { \<A> <http://dbpedia.org/ontology/address> ?x }" and GQT is "SELECT ?a WHERE { ?a <http://dbpedia.org/ontology/address> [] . ?a a <http://dbpedia.org/ontology/Place> }"
6) **Final Formatting of the SQT and the GQT:** The queries were processed i.e lower, etc.
7) **Generator Functionality:** The templates prepared in the previous step were used to fetch examples using sparql endpoint. Using this step, the data.en and the data.sparql file were created. For instance, an example from data.en is what is the address of kauffman stadium, and of data.sparql is select var_x where brack_open dbr_Kauffman_Stadium <dbo_address> var_x brack_close
8) **Vocabulary Building:** Separate vocabulary files were created using data.en and data.sparql
9) **Train_dev_test Split:** The data.en and data.sparql were split into 3 files respectively.
10) **Train:** nmt repo from tensorlfow was forked using git submodules and was used to train the model using train.sh file. But as files contained a lot of tf functions that were deprecated, I decided to not train the models rather work with the latest learner module.