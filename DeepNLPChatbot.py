############################################################## Deep NLP Chatbot #################################################


###Importing Libraries
import numpy as np
import tensorflow as tf
import re
import time



############################################################ PART 1 - DATA PREPROCESSING ########################################



# Importing the dataset https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

## Actual conversations       # avoid import error
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

## list of converstions per movie
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')



# Dictionary that maps each movie line with its id #############################

#FROM:
#  1           2         3          4           5   
#L254 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ If I was Bianca, it would be, "Any school you want, precious.  Don't forget your tiara."
#L253 +++$+++ u6 +++$+++ m0 +++$+++ MANDELLA +++$+++ Does it matter?
#L252 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ I appreciate your efforts toward a speedy death, but I'm consuming.  Do you mind?
#L251 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ Neither has his heterosexuality.
#L250 +++$+++ u6 +++$+++ m0 +++$+++ MANDELLA +++$+++ That's never been proven   

#TO: 

#L10000 {Oh... chamber runs.  Uh huh, that's good.  Well, hey... you guys know any songs?}



id2line = {}
for line in lines:
        _line = line.split(' +++$+++ ')                    ## _ for local variable in the loop
        if len(_line) == 5:                                ## Make sure that line has 5 elements to avoid shifting issue
                id2line[_line[0]] = _line[4]               ## maps each sentence with id



# Creating a list of conversations #############################################

#FROM:
# u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']     
                
#TO:                
# ['L194', 'L195', 'L196', 'L197']                
conversations_ids = []
for conversation in conversations[:-1]:
###                                              remove sqare brackets | remove quotes | remove empty space    
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")  
        conversations_ids.append(_conversation.split(','))
        
        
        
# Separating Questions & Answers. Making sure that lists are same size #########

questions = []
answers = []
for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
                questions.append(id2line[conversation[i]])  ## first line to questions list
                answers.append(id2line[conversation[i+1]])  ## second line to answers list
               
                
                
# Text Cleaning - Step 1########################################################
 
def clean_text(text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", " will not", text)
        text = re.sub(r"can't", " cannot", text)
        text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
        return text



# Cleaning Questions & Answers #################################################
        
clean_questions = []
for question in questions:
        clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
        clean_answers.append(clean_text(answer))
   

             
# Dictionary of word occurences ################################################
 
### (removing rare words for performance optimizing)    
word2count = {}
for question in clean_questions:
        for word in question.split():
                if word not in word2count:
                        word2count[word] = 1
                else:
                        word2count[word] += 1
for answer in clean_answers:
        for word in answer.split():
                if word not in word2count:
                        word2count[word] = 1
                else:
                        word2count[word] += 1

### Filtering out less occurent words and transform remaining ones into integers

threshold = 20  ## number of occurences in dictionary
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
        if count >= threshold:
                questionswords2int[word] = word_number
                word_number += 1
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
        if count >= threshold:
                answerswords2int[word] = word_number
                word_number += 1
                
### Adding NLP tokens for two dictionaries

## SOS - start of sentence
## EOS - end of sentence
## PAD - fill empty space to match length of sentence
## OUT - words that were filtered out previously by questionswords2int and answerswords2int              


tokens = ['<PAD>','<EOS>','<OUT>','<SOS>'] 
for token in tokens:
        questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
        answerswords2int[token] = len(answerswords2int) + 1       

### Creating the inverse dictionary for answerswords2int for seq2seq implementation
#             word integers: word for word, word integers in previous dictionary
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}    

### Adding EOS token to the end of the answers

for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>' 
        
        
### Translating all the questions and the answers into integers and filtering <OUT> non-frequent words.

questions_into_int = []
for question in clean_questions:
        ints = []
        for word in question.split():
                if word not in questionswords2int:
                        ints.append(questionswords2int['<OUT>'])
                else:
                        ints.append(questionswords2int[word])
        questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
        ints = []
        for word in answer.split():
                if word not in answerswords2int:
                        ints.append(answerswords2int['<OUT>'])
                else:
                        ints.append(answerswords2int[word])
        answers_into_int.append(ints)

### Sorting questions and answers by the length of questions (Speed-up the training and loss reduction)

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
        for i in enumerate(questions_into_int):
                if len(i[1]) == length:
                        sorted_clean_questions.append(questions_into_int[i[0]])
                        sorted_clean_answers.append(answers_into_int[i[0]])