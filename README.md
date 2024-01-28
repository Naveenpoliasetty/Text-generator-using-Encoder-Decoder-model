# Text-generator-using-Encoder-Decoder-model
## Prequisites
## What is LSTM ?
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture, designed to overcome the vanishing gradient problem and capture long-term dependencies in sequential data. LSTMs were introduced by Sepp Hochreiter and Jürgen Schmidhuber in 1997 and have since become a popular choice for tasks involving sequential information, such as natural language processing, speech recognition, and time series analysis.

Features of LSTMs:

Memory Cells:

LSTMs have memory cells that can store information for long durations. This helps in capturing dependencies over extended time steps.
Gates:

LSTMs use three gates to control the flow of information:
Forget Gate: Decides what information to throw away from the cell state.
Input Gate: Modifies the cell state by adding new information.
Output Gate: Decides what the next hidden state should be based on the cell state.

Cell State:
The cell state runs throughout the entire sequence, allowing the model to remember information over long periods.

Hidden State:
The hidden state is responsible for carrying information that is deemed relevant for making predictions based on the current input and past information.

Training and Backpropagation:
LSTMs use backpropagation through time (BPTT) for training, addressing the vanishing gradient problem that traditional RNNs often face.

Bidirectional LSTMs:
LSTMs can be used in a bidirectional manner, processing input sequences in both forward and backward directions. This helps capture dependencies from both past and future contexts.

## Working

#### Text preprocessing
The process involves iterating over each line in the input text, assuming sentences are separated by '\n'. Each sentence is tokenized using the `texts_to_sequences` method of a tokenizer (`tk`), converting words into numerical indices based on the vocabulary learned during fitting. Sequences of increasing lengths are then created for each tokenized sentence, starting from the first word up to the entire sentence, and appended to a list named `sequences`. Additionally, the `pad_sequences` function from `tensorflow.keras.utils` is imported to ensure all sequences have the same length. This function is then applied to the list of sequences (`squence`), padding or truncating them as needed and storing the result in a variable named `input_sequence`.

For feature extraction, variable `X` is created by selecting all rows and all columns except the last one from `input_sequence`, excluding the last element in each sequence. For target label extraction, variable `y` is created by selecting all rows and only the last column from `input_sequence`, representing the next element in each sequence.

To prepare target labels (`y`) for training, they are one-hot encoded using the `to_categorical` function. The assumption is made that there are 255 classes, and the encoding is designed to cover the range from 0 to 254.


### Implementing Neural Network:

Import the necessary modules from TensorFlow Keras, including Sequential, Dense, Embedding, LSTM, and Bidirectional.
Creating a Sequential Model (model):

Create an instance of the Sequential model named model.
Adding an Embedding Layer:

Add an Embedding layer to the model.
The vocabulary size is set to 255 (input_dim=255).
The embedding dimension is set to 100 (output_dim=100).
The input length is set to 23 (input_length=23), corresponding to the length of each input sequence.
Adding a Bidirectional LSTM Layer:

Add a Bidirectional layer wrapping an LSTM layer to the model.
The LSTM layer has 150 units.
Adding a Dense Output Layer:

Add a Dense layer to the model for the output layer.
The number of units is set to 255, matching the number of classes.
The activation function is set to 'softmax' for multi-class classification.
Compiling the Model:

Compile the model using categorical crossentropy as the loss function (loss='categorical_crossentropy'), the Adam optimizer (optimizer='adam'), and accuracy as the metric (metrics=['accuracy']).
Training the Model:

Train the model using the fit method.
X is the input data, and y is the target data.
The number of training epochs is set to 100 (epochs=100).

### Text generation using the implemented model

The `word_predictor` function predicts the next words in a given text using a trained model. Here's how it works:

1. Tokenization and Padding:
   - Tokenize the input text and pad the sequence with zeros to match the model's input length.

2. Model Prediction:
   - Use the trained model to predict the next word in the sequence.

3. Word Lookup:
   - Map the predicted index to an actual word using the tokenizer's vocabulary.

4. Repeat:
   - Repeat these steps for the desired number of words (`len`).

5. Printing Result:
   - Print the final text with the predicted words.

This function generates a sequence of predicted words based on the input text and the trained language model. The quality of predictions depends on the effectiveness of the model and tokenizer.
