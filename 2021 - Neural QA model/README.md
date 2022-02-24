The idea of the project is to increase the data using syntax aware method and backtranslation which will ultimately extend the coverage and at the same time make the model more real and efficient. 
I tried to implement them steps in order to gain a better understanding of the project by reading the bog post by Siddhant Jain- https://imsiddhant07.github.io/Neural-QA-Model-for-DBpedia/ 
and the entire code can be found here- https://github.com/dbpedia/neural-qa/tree/gsoc-siddhant

1) Syntax-aware: Here, two techniques are used. But before that all the words in the english sentence are assigned a probability using graph methods 
i.e parsing tree. Then the words are ordered. Data is then created by dropping the top words or by replacing them top words by its synonyms.

2) BackTranslation: In order to make the model robust, sp-en model is created which unlike previous years considers sparql input and predicts an english question. 
These english questions are then added to the training dataset with its corresponding sparql query for training purposes.

For experimentation purposes, the **Annotations_F30_art.csv** dataset was used.
<hr>
The training dataset can be found here- https://github.com/sauravjoshi23/NMT/tree/main/2021%20-%20Neural%20QA%20model/DataAug/Data/art_30 <br>
<hr>
The syntax aware output can be found here- https://github.com/sauravjoshi23/NMT/tree/main/2021%20-%20Neural%20QA%20model/DataAug/Data/art_30/Syntax-aware <br>
<hr>
The backtranslation output can be found here- https://github.com/sauravjoshi23/NMT/tree/main/2021%20-%20Neural%20QA%20model/DataAug/Data/art_30/BackTranslation <br>
