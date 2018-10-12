#Objective:
In the software testing field, while a tester would attempt to raise a new defect/bug, using this cognitive approach
the tester would be able to know whether a similar defect is already available in the existing defect pool and how much
similar the new defect is with existing the defects by comparing similarity score calculated by this tool

#Approach:
1.Extract existing defects(Number of defects in the existing defect pool : 1000+)
2.Create a data matrix containg defect id, defect summary and defect description
3.Clean and tokenise the defect description part of each defect
4.Vectorise the tokens using word2vec model in gensim package
5.Calculate the average vector for all words in every document(defect description in this case) 
6.Similarly calculate average vector for the description of the new defect
7.Next calculate cosine similarity between the average vectors

#DataSet:
For presentation purpose Quora questionnaire from Kaggle has been used here

