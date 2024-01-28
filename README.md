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
Import Libraries:
Import TensorFlow and the Tokenizer class from Keras.

Define Input Text:<br>
Create a variable named data that holds your input text. This could be any text data you want to process.

Initialize Tokenizer:<br>
Create an instance of the Tokenizer class called tk.
Fit the tokenizer on the input text using the fit_on_texts method. This step builds a vocabulary based on the words in the text.

Tokenization Loop:<br>
Iterate over each line in the input text (assuming sentences are separated by '\n').
Tokenize each sentence using the texts_to_sequences method of the tokenizer (tk). This method converts each word in the sentence to a numerical index based on the vocabulary learned during fitting.

Create Sequences:<br>
For each tokenized sentence, create sequences of increasing lengths, starting from the first word up to the entire sentence.
Append these sequences to the sequences list.

Import Pad Sequences:<br>
Import the pad_sequences function from tensorflow.keras.utils. This function is used to ensure that all sequences in a list have the same length by padding or truncating them as needed.

Pad Sequences:<br>
Create a variable named input_sequence.
Use pad_sequences to pad the list of sequences (squence) with zeros to ensure they all have the same length.
maxlen is the maximum length of the sequences after padding.
padding='pre' specifies that the padding should be added to the beginning of each sequence.


Extracting Input Features (X):<br>

Create a variable named X.
Select all rows and all columns except the last one from the input_sequence using input_squence[:,:-1]. This is done to create input features for the model, excluding the last element in each sequence.
Display the shape of the resulting X array using X.shape.

Extracting Target Labels (y):<br>

Create a variable named y.
Select all rows and only the last column from the input_sequence using input_squence[:,-1]. This is done to create target labels for the model, representing the next element in each sequence.
Display the shape of the resulting y array using y.shape.

One-Hot Encoding Target Labels (y):<br>

Import the to_categorical function from tensorflow.keras.utils.
Use to_categorical to convert the target labels (y) into one-hot encoded format.
num_classes=255 specifies the number of classes for one-hot encoding. It assumes that the target labels range from 0 to 254.

Importing Necessary Modules:

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
