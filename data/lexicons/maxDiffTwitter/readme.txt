SemEval-2015 English Twitter Lexicon
Version 1.0
12 April 2016
Copyright (C) 2016 National Research Council Canada (NRC)
Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


***************************************
Terms of use
***************************************
1. This lexicon can be used freely for research purposes. 
2. The papers listed below provide details of the creation and use of the lexicon. If you use a lexicon, then please cite the associated papers.
3. If interested in commercial use of the lexicon, send email to the contact. 
4. If you use the lexicon in a product or application, then please credit the authors and NRC appropriately. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.
5. National Research Council Canada (NRC) disclaims any responsibility for the use of the lexicon and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.
6. Rather than redistributing the data, please direct interested parties to this page:
   http://www.saifmohammad.com/WebPages/SCL.html


Please feel free to send us an email:
- with feedback regarding the lexicon; 
- with information on how you have used the lexicon;
- if interested in having us analyze your data for sentiment, emotion, and other affectual information;
- if interested in a collaborative research project.


***************************************
General Description
***************************************

SemEval-2015 English Twitter Lexicon is a list of single words and simple two-word negated expressions and their associations with positive and negative sentiment. The terms are drawn from English Twitter and include general English words, misspellings, hashtags, and other categories frequently used in Twitter. The negated expressions are in the form of 'negator w', where 'negator' is a negation trigger. The complete list of the negation triggers is included into this distribution.

The sentiment associations were obtained manually through crowdsourcing using the Best-Worst Scaling annotation technique.


***************************************
SemEval-2015
***************************************

This lexicon was used in SemEval-2015 shared task on Sentiment Analysis in Twitter (Task 10 Subtask E) (http://alt.qcri.org/semeval2015/task10/). The sentiment association scores were converted into the range 0..1 for the SemEval competition. 



***************************************
File Format
***************************************

Each line in the file has the following format:

<sentiment score><tab><term>

where
<sentiment score> is a real number between -1 and 1 indicating the degree of association of the term with positive sentiment;
<term> is a single word or a multi-word phrase.

There are 1,515 terms.


***************************************
More Information
***************************************

For more details, refer to Section 6.1.2 in
	Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts.  Journal of Artificial Intelligence Research, 50:723-762, 2014.




