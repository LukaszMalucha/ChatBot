############################################################## Deep NLP Chatbot #################################################


###Importing Libraries
import numpy as np
import tensorflow as tf
import re
import time

import io




############################################################ PART 1 - DATA PREPROCESSING ########################################



# Importing the dataset https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

## Actual conversations       # avoid import error
with io.open('movie_lines.txt', encoding = 'utf-8', errors='ignore') as source:
    lines = source.read().split('\n')

## list of converstions per movie
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
with io.open('movie_conversations.txt', encoding = 'utf-8', errors='ignore') as source:
    conversations = source.read().split('\n')


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
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

    
# Cleaning Questions & Answers #################################################
        
clean_questions = []
for question in questions:
        clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
        clean_answers.append(clean_text(answer))
   

# Filtering out the questions and answers that are too short or too long #######
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
    
             
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

threshold = 15  ## number of occurences in dictionary
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
        
        
#EFFECT:
# [862, 1138, 7542, 5197, 3042, 3811, 3797, 8826, 8428, 2425, 8181, 5573, 3797, 109, 8826]        



### Sorting questions and answers by the length of questions (Speed-up the training and loss reduction)

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
        for i in enumerate(questions_into_int):
                if len(i[1]) == length:
                        sorted_clean_questions.append(questions_into_int[i[0]])
                        sorted_clean_answers.append(answers_into_int[i[0]])
      
#############################
                        ##############
                        ###########
sorted_clean_questions = sorted_clean_questions[:1000]                       
sorted_clean_answers = sorted_clean_answers[:1000]                        
######################################################### PART 2 - BUILDING THE SEQ2SEQ MODEL ######################################        

### Creating placeholders for the inputs and the targets 

def model_inputs():
        inputs = tf.placeholder(tf.int32, [None, None], name = 'input')        ## Question placeholder tensor, 2d - list of lists
        targets = tf.placeholder(tf.int32, [None, None], name = 'target')      ## Answer placeholder tensor, 2d - list of lists
        lr = tf.placeholder(tf.float32, name = 'learning_rate')                ## Learn rate 
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')             ## Dropout rate
        return inputs, targets, lr, keep_prob



### Preprocessing the targets - turn into acceptable format of a target(answers)
        
                          # maps tokens to ints    
def preprocess_targets(targets, word2int, batch_size):
        left_side = tf.fill([batch_size, 1], word2int['<SOS>'])                ## matrix of 10 rows and one column filled with SOS
                                             ## start end exc. last one  slide      
        right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1]) ## 10 elements of each row without last token EOS
        preprocessed_targets = tf.concat([left_side, right_side], 1)           ## horizontal concatination
        return preprocessed_targets



### Creating the Encoder RNN Layer
                                                           # list of the length of the question                                            
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)                                         ## init lstm cell
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)       ## init dropout 
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)               ## init encoder cell lstm dropout layer times num_layers
        ## _, - we only need second element
        _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                           cell_bw = encoder_cell,
                                                           sequence_length = sequence_length, ## list of the length of the question
                                                           inputs = rnn_inputs,
                                                           dtype = tf.float32)
        return encoder_state



### Decoding the training set

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope,
                        output_function, keep_prob, batch_size):
        attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option = 'bahdanau',
                                                                                                                                        num_units = decoder_cell.output_size)
        training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                                  attention_keys,
                                                                                  attention_values,
                                                                                  attention_score_function,
                                                                                  attention_construct_function,
                                                                                  name = "attn_dec_train")
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                      training_decoder_function,
                                                                      decoder_embedded_input,
                                                                      sequence_length,
                                                                      scope = decoding_scope)
        decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
        return output_function(decoder_output_dropout)




### Decoding the test set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
        attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option = 'bahdanau',
                                                                                                                                        num_units = decoder_cell.output_size)
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                  encoder_state[0],
                                                                                  attention_keys,
                                                                                  attention_values,
                                                                                  attention_score_function,
                                                                                  attention_construct_function,
                                                                                  decoder_embeddings_matrix, 
                                                                                  sos_id, 
                                                                                  eos_id, 
                                                                                  maximum_length, 
                                                                                  num_words, 
                                                                                  name = "attn_dec_inf")
        test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                        test_decoder_function,
                                                                        scope = decoding_scope)
        return test_predictions




### Creating Decoder RNN

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, 
                sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
        with tf.variable_scope("decoding") as decoding_scope:
                lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
                lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
                decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
                weights = tf.truncated_normal_initializer(stddev = 0.1)
                biases = tf.zeros_initializer()
                output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                              num_words,
                                                                              None,
                                                                              scope = decoding_scope,
                                                                              weights_initializer = weights,
                                                                              biases_initializer = biases)
                training_predictions = decode_training_set(encoder_state,
                                                           decoder_cell,
                                                           decoder_embedded_input,
                                                           sequence_length,
                                                           decoding_scope,
                                                           output_function,
                                                           keep_prob,
                                                           batch_size)
                decoding_scope.reuse_variables()
                test_predictions = decode_test_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embeddings_matrix,
                                                   word2int['<SOS>'],
                                                   word2int['<EOS>'],
                                                   sequence_length -1,
                                                   num_words,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        return training_predictions, test_predictions 



### Building SEQ2SEQ model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, 
                  answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, 
                  rnn_size, num_layers, questionswords2int):
        encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                                  answers_num_words +1,
                                                                  encoder_embedding_size,
                                                                  initializer = tf.random_uniform_initializer(0,1))
        encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
        preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
        decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
        training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                             decoder_embeddings_matrix,
                                                             encoder_state,
                                                             questions_num_words,
                                                             sequence_length,
                                                             rnn_size,
                                                             num_layers,
                                                             questionswords2int,
                                                             keep_prob,
                                                             batch_size)
        return training_predictions, test_predictions



############################################################ PART 3 - TRAINING THE SEQ2SEQ MODEL ##############################################

### Setting the Hyperparameters

#epochs = 1
#batch_size = 32
#rnn_size = 1024
#num_layers = 3
#encoding_embedding_size = 512
#decoding_embedding_size = 512
#learning_rate = 0.001
#learning_rate_decay = 0.9
#min_learning_rate = 0.0001
#keep_probability = 0.5

epochs = 1                   ## 100
batch_size = 64            ## 64
rnn_size = 5               ## 512   
num_layers = 1              # 3
encoding_embedding_size = 5
decoding_embedding_size = 5
learning_rate = 1       ## 0.01
learning_rate_decay = 1 ## 0.9
min_learning_rate = 0.1
keep_probability = 0.5

### Defining Session (reset graph first)

tf.reset_default_graph()
session = tf.InteractiveSession()

### Loading the model inputs into our function

inputs, targets, lr, keep_prob = model_inputs()

### Setting the sequence length to max 25 length

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
    
### Getting the shape of the inputs tensor    

input_shape = tf.shape(inputs)  

### Getting the training and test predictions 
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size, 
                                                       num_layers,
                                                       questionswords2int)
        
### Setting up the Loss Error, the Optimizer and Gradient Clipping

with tf.name_scope('optimization'): 
        loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, 
                                                      targets, 
                                                      tf.ones([input_shape[0],
                                                      sequence_length]))               
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss_error)
        clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.,5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
        optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
                       
### Padding the sequences with <PAD> to equalize lengths of Q & A
def apply_padding(batch_of_sequences, word2int):
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
        return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

        
### Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
        
        for batch_index in range(0, len(questions) // batch_size):
                start_index = batch_index * batch_size
                questions_in_batch = questions[start_index : start_index + batch_size]
                answers_in_batch = answers[start_index : start_index + batch_size]
                padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
                padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
                yield padded_questions_in_batch, padded_answers_in_batch 
                
### Splitting the questions and answers into training and validation sets
                
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]     

### TRAINING

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "./chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"



session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
        
            for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
                starting_time = time.time()
                _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                                       targets: padded_answers_in_batch,
                                                                                                       lr: learning_rate,
                                                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                                                       keep_prob: keep_probability})
                total_training_loss_error += batch_training_loss_error
                ending_time = time.time()
                batch_time = ending_time - starting_time
                if batch_index % batch_index_check_training_loss == 0:
                    print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                               epochs,
                                                                                                                                               batch_index,
                                                                                                                                               len(training_questions) // batch_size,
                                                                                                                                               total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
                    total_training_loss_error = 0
                if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
                    total_validation_loss_error = 0
                    starting_time = time.time()
                    for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                        batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                               targets: padded_answers_in_batch,
                                                                               lr: learning_rate,
                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                               keep_prob: 1})
                        total_validation_loss_error += batch_validation_loss_error
                    ending_time = time.time()
                    batch_time = ending_time - starting_time
                    average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
                    print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
                    learning_rate *= learning_rate_decay
                    if learning_rate < min_learning_rate:
                        learning_rate = min_learning_rate
                    list_validation_loss_error.append(average_validation_loss_error)
                    if average_validation_loss_error <= min(list_validation_loss_error):
                        print('I speak better now!!')
                        early_stopping_check = 0
                        saver = tf.train.Saver()
                        saver.save(session, checkpoint)
                    else:
                        print("Sorry I do not speak better, I need to practice more.")
                        early_stopping_check += 1
                        if early_stopping_check == early_stopping_stop:
                            break
            if early_stopping_check == early_stopping_stop:
                print("My apologies, I cannot speak better anymore. This is the best I can do.")
                break
print("Game Over")          
                        
        
######################################################## PART 4 - TESTING THE SEQ2SEQ MODEL ############################################        
        
        
# ### Loading weights and running session
# checkpoint = './chatbot_weights.ckpt'
# session = tf.InteractiveSession()
# session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(session, checkpoint)        

# ### Converting the questions from strings to list of encoding integers
# def convert_string2int(question, word2int):
#         question = clean_text(question)
#         return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
        
# ### Setting up Chat
# while(True):
#     question = input("You: ")
#     if question == 'Goodbye':
#         break
#     question = convert_string2int(question, questionswords2int)
#     question = question + [questionswords2int['<PAD>']] * (25 - len(question))
#     fake_batch = np.zeros((batch_size, 25))
#     fake_batch[0] = question
#     predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
#     answer = ''
#     for i in np.argmax(predicted_answer, 1):
#         if answersint2word[i] == 'i':
#             token = ' I'
#         elif answersint2word[i] == '<EOS>':
#             token = '.'
#         elif answersint2word[i] == '<OUT>':
#             token = 'out'
#         elif answersint2word[i] == '<PAD>':   
#             token = ''
#         else:
#             token = ' ' + answersint2word[i]
#         answer += token
#         if token == '.':
#             break
#     print('ChatBot: ' + answer)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
             
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        