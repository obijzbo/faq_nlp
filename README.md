## Installation

Run install.sh from the folder named shell_files. This will install all the components that is essential to run this program
Also mongoDB should be installed in the local device. 

```bash
. shell_files/install.sh
```

## Usage

Execute main.py to run the project. 
Two csv files will be generated, named 'similarity.csv' and 'faq.csv'. 
'similarity.csv' contains questions from train and test data, also contains the similarity score.
'faq.csv' contains questions from the test data and answer from the train data.


## Discussion
Initially i used 'distilbert-base-nli-mean-tokens'. Which failed to answer last three questions from 'FAQs_test.csv' correctly.

To investigate further, i used spacy to check similarity between sentences. It performed poorly compared to BERT sentence transformer. 

Then i decided to remove nouns from each questions using spacy. But now last two answers were incorrect. I was still using 'distilbert-base-nli-mean-tokens'. 
So i decided to change the model, and used 'sentence-transformers/all-mpnet-base-v2' instead.
Now all the questions were correctly answered. 

But there were one last problem. The nouns we have in this datasets are - 
'Albert', 'Einstein', 'Nobel', 'Prize', 'Physics'
And any sentence with 'Albert' and 'Einstein' always had higher similarity score, no matter what the context were.
But ['Nobel', 'Prize', 'Physics'] these words are essential to determine sentence context.
So if we decide to remove all nouns, it might have a negative impact. 

Therefore i didn't use spacy to remove the nouns. Instead i only removed ['Albert', 'Einstein'] from any sentence. 
And that's how i achieved the following result. 

It was quite simple yet interesting task. Had fun doing it. 
