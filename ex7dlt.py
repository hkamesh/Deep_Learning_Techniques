import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_texts = [
    'hello', 'how are you', 'what is your name', 'bye', 'what can you do',
    'tell me a joke', 'thank you', 'good morning', 'good night'
]

target_texts = [
    'hi', 'i am fine', 'i am a bot', 'goodbye', 'i can chat with you',
    'why did the chicken cross the road', 'you are welcome', 'morning', 'night'
]

target_texts = ['<start> ' + txt + ' <end>' for txt in target_texts]

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

input_sequences = pad_sequences(input_sequences, padding='post')
target_sequences = pad_sequences(target_sequences, padding='post')

vocab_size = len(tokenizer.word_index) + 1
max_input_len = input_sequences.shape[1]
max_target_len = target_sequences.shape[1]

embedding_dim = 64
lstm_units = 128

encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(enc_emb)

decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm_outputs, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb, initial_state=[state_h, state_c]
)

attention = Attention()
context_vector = attention([decoder_lstm_outputs, encoder_outputs])
concat = Concatenate(axis=-1)([context_vector, decoder_lstm_outputs])
decoder_outputs = Dense(vocab_size, activation='softmax')(concat)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

decoder_target_data = np.expand_dims(target_sequences, -1)

model.fit([input_sequences, target_sequences], decoder_target_data, batch_size=2, epochs=300, verbose=0)

encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

dec_state_input_h = Input(shape=(lstm_units,))
dec_state_input_c = Input(shape=(lstm_units,))
enc_output_input = Input(shape=(max_input_len, lstm_units))

dec_emb2 = dec_emb_layer(decoder_inputs)
dec_lstm_outputs2, state_h2, state_c2 = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c]
)

context_vector2 = attention([dec_lstm_outputs2, enc_output_input])
concat2 = Concatenate(axis=-1)([context_vector2, dec_lstm_outputs2])
dec_outputs2 = Dense(vocab_size, activation='softmax')(concat2)

decoder_model = Model(
    [decoder_inputs, enc_output_input, dec_state_input_h, dec_state_input_c],
    [dec_outputs2, state_h2, state_c2]
)

def generate_response(input_text, temperature=1.0):
    def sample_with_temperature(probabilities, temperature=1.0):
        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = np.log(probabilities + 1e-10) / temperature
        exp_probs = np.exp(probabilities)
        probabilities = exp_probs / np.sum(exp_probs)
        return np.random.choice(len(probabilities), p=probabilities)

    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    enc_out, state_h, state_c = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.array([[tokenizer.word_index['<start>']]])
    decoded_sentence = ''
    prev_word = ''

    for _ in range(max_target_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq, enc_out, state_h, state_c], verbose=0
        )
        sampled_token_index = sample_with_temperature(output_tokens[0, -1, :], temperature)
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word in ('<end>', '', prev_word):
            break

        decoded_sentence += sampled_word + ' '
        prev_word = sampled_word

        target_seq = np.array([[sampled_token_index]])
        state_h, state_c = h, c

    return decoded_sentence.strip()

test_inputs = [
    "hello",
    "how are you",
    "what is your name",
    "bye",
    "tell me a joke",
    "what can you do",
    "good night"
]

for sentence in test_inputs:
    print("Input:", sentence)
    print("Bot  :", generate_response(sentence, temperature=0.8))
    print('-' * 30)
