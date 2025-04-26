import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, TimeDistributed
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from nltk.translate.bleu_score import sentence_bleu
import os

# 1. Pengambilan Data dari NewsAPI
def fetch_news_data(api_key, query='technology', max_articles=1000):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json()['articles'][:max_articles]

    data = []
    for article in articles:
        if article['title'] and article['content']:
            data.append({
                'title': article['title'],
                'content': article['content']
            })
    return pd.DataFrame(data)

# 2. Preprocessing
def preprocess_data(df, max_words=20000, max_len_content=200, max_len_title=20):
    # Add start and end tokens to titles
    df['title'] = df['title'].apply(lambda x: '<start> ' + x + ' <end>')

    # Tokenisasi
    content_tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    title_tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    content_tokenizer.fit_on_texts(df['content'])
    title_tokenizer.fit_on_texts(df['title'])

    # Ensure vocabulary sizes are sufficient
    content_vocab_size = min(max_words, len(content_tokenizer.word_index) + 1)
    title_vocab_size = min(max_words, len(title_tokenizer.word_index) + 1)

    # Konversi ke sequence
    content_seq = content_tokenizer.texts_to_sequences(df['content'])
    title_seq = title_tokenizer.texts_to_sequences(df['title'])

    # Ensure no index exceeds vocabulary size
    content_seq = [[idx if idx < content_vocab_size else 0 for idx in seq] for seq in content_seq]
    title_seq = [[idx if idx < title_vocab_size else 0 for idx in seq] for seq in title_seq]

    # Padding
    content_padded = pad_sequences(content_seq, maxlen=max_len_content, padding='post')
    title_padded = pad_sequences(title_seq, maxlen=max_len_title, padding='post')

    return content_padded, title_padded, content_tokenizer, title_tokenizer, content_vocab_size, title_vocab_size

# 3. Model 1: Encoder-Decoder Dasar dengan LSTM
def build_lstm_model(content_vocab_size, title_vocab_size, embedding_dim=256, lstm_units=512, max_len_title=20):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(content_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True)
    _, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb_layer = Embedding(title_vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(title_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define inference models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

# 4. Model 2: Encoder-Decoder with Attention
def build_attention_model(content_vocab_size, title_vocab_size, embedding_dim=256, lstm_units=512, max_len_title=20):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(content_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb_layer = Embedding(title_vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Attention layer
    attention = MultiHeadAttention(num_heads=4, key_dim=lstm_units//4)
    context_vector = attention(decoder_outputs, encoder_outputs)

    # Combine attention with decoder output
    decoder_combined_context = Concatenate()([decoder_outputs, context_vector])

    # Output layer
    decoder_dense = Dense(title_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_combined_context)

    # Define model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

# 5. Model 3: Transformer
def build_transformer_model(content_vocab_size, title_vocab_size, max_len_content=200, max_len_title=20, embedding_dim=256):
    # Encoder
    encoder_inputs = Input(shape=(max_len_content,), name='encoder_inputs')
    enc_embedding = Embedding(content_vocab_size, embedding_dim)(encoder_inputs)
    enc_embedding = enc_embedding + tf.keras.layers.Embedding(max_len_content, embedding_dim, trainable=True)(tf.range(start=0, limit=max_len_content, delta=1))
    enc_outputs = enc_embedding

    # Encoder blocks
    for _ in range(2):
        # Self attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim//4)(enc_outputs, enc_outputs)
        attn_output = Dropout(0.1)(attn_output)
        enc_outputs = LayerNormalization(epsilon=1e-6)(enc_outputs + attn_output)

        # Feed forward
        ffn_output = Dense(512, activation='relu')(enc_outputs)
        ffn_output = Dense(embedding_dim)(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        enc_outputs = LayerNormalization(epsilon=1e-6)(enc_outputs + ffn_output)

    # Decoder
    decoder_inputs = Input(shape=(max_len_title,), name='decoder_inputs')
    dec_embedding = Embedding(title_vocab_size, embedding_dim)(decoder_inputs)
    dec_embedding = dec_embedding + tf.keras.layers.Embedding(max_len_title, embedding_dim, trainable=True)(tf.range(start=0, limit=max_len_title, delta=1))
    dec_outputs = dec_embedding

    # Decoder blocks
    for _ in range(2):
        # Self attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim//4)(dec_outputs, dec_outputs)
        attn_output = Dropout(0.1)(attn_output)
        dec_outputs = LayerNormalization(epsilon=1e-6)(dec_outputs + attn_output)

        # Cross attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=embedding_dim//4)(dec_outputs, enc_outputs)
        attn_output = Dropout(0.1)(attn_output)
        dec_outputs = LayerNormalization(epsilon=1e-6)(dec_outputs + attn_output)

        # Feed forward
        ffn_output = Dense(512, activation='relu')(dec_outputs)
        ffn_output = Dense(embedding_dim)(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        dec_outputs = LayerNormalization(epsilon=1e-6)(dec_outputs + ffn_output)

    # Output layer
    outputs = Dense(title_vocab_size, activation='softmax')(dec_outputs)

    # Define model
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model

# 6. Evaluate generated titles
def evaluate_titles(generated_titles, reference_titles):
    bleu_scores = []

    for gen, ref in zip(generated_titles, reference_titles):
        # BLEU
        bleu = sentence_bleu([ref.split()], gen.split())
        bleu_scores.append(bleu)

    return {
        'avg_bleu': np.mean(bleu_scores)
    }

# 7. Generate title function for LSTM with inference model
def generate_title_lstm(encoder_model, decoder_model, content, content_tokenizer, title_tokenizer, title_vocab_size, max_len_title=20):
    # Encode input sequence
    content_seq = content_tokenizer.texts_to_sequences([content])
    # Ensure no index exceeds vocabulary size
    content_seq = [[idx if idx < title_vocab_size else 0 for idx in seq] for seq in content_seq]
    content_padded = pad_sequences(content_seq, maxlen=200, padding='post')
    states_value = encoder_model.predict(content_padded, verbose=0)

    # Generate title
    target_seq = np.zeros((1, 1))
    start_token = title_tokenizer.word_index.get('<start>', 1)
    if start_token >= title_vocab_size:
        start_token = 1  # Use a safe token if <start> is out of range
    target_seq[0, 0] = start_token

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index >= title_vocab_size:
            sampled_token_index = 0  # Use padding token if out of range
        sampled_word = title_tokenizer.index_word.get(sampled_token_index, '')

        end_token = title_tokenizer.word_index.get('<end>', 2)
        if end_token >= title_vocab_size:
            end_token = 2  # Use a safe token if <end> is out of range

        if sampled_word == '<end>' or sampled_token_index == end_token or len(decoded_sentence.split()) >= max_len_title - 1:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '

        # Update target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# 8. Generate title for other models
def generate_title(model, content, content_tokenizer, title_tokenizer, content_vocab_size, title_vocab_size, max_len_title=20):
    # Encode input sequence
    content_seq = content_tokenizer.texts_to_sequences([content])
    # Ensure no index exceeds vocabulary size
    content_seq = [[idx if idx < content_vocab_size else 0 for idx in seq] for seq in content_seq]
    content_padded = pad_sequences(content_seq, maxlen=200, padding='post')

    # Start with <start> token
    target_seq = np.zeros((1, max_len_title))
    start_token = title_tokenizer.word_index.get('<start>', 1)
    if start_token >= title_vocab_size:
        start_token = 1  # Use a safe token if <start> is out of range
    target_seq[0, 0] = start_token

    generated_title = []

    for i in range(1, max_len_title):
        # Predict next token
        predictions = model.predict([content_padded, target_seq], verbose=0)

        # Sample token
        predicted_id = np.argmax(predictions[0, i-1])
        if predicted_id >= title_vocab_size:
            predicted_id = 0  # Use padding token if out of range

        end_token = title_tokenizer.word_index.get('<end>', 2)
        if end_token >= title_vocab_size:
            end_token = 2  # Use a safe token if <end> is out of range

        # If end token or max length, stop
        if predicted_id == end_token or predicted_id == title_tokenizer.word_index.get('<end>', 0) or i == max_len_title-1:
            break

        # Add token to sequence
        generated_title.append(title_tokenizer.index_word.get(predicted_id, ''))
        target_seq[0, i] = predicted_id

    return ' '.join(generated_title)

# 9. Main Execution
def main():
    # Sample data if API key is not available
    sample_data = pd.DataFrame({
        'title': ['Technology transforms healthcare', 'AI breakthroughs in 2024'],
        'content': ['The healthcare industry is being transformed by new technologies...',
                   'Artificial intelligence has seen major breakthroughs in 2024...']
    })

    # Try to get API data, fallback to sample data
    try:
        # Replace with your API key
        api_key = '44dfd24144424d83bebfd58813ab6cf7'
        df = fetch_news_data(api_key)
        if len(df) < 10:  # If not enough data retrieved
            df = sample_data
    except:
        print("Using sample data instead of API data")
        df = sample_data

    # Print dataset size
    print(f"Dataset size: {len(df)} articles")

    # Preprocessing with increased vocabulary size
    content_padded, title_padded, content_tokenizer, title_tokenizer, content_vocab_size, title_vocab_size = preprocess_data(df, max_words=50000)

    # Print vocabulary sizes
    print(f"Content vocabulary size: {content_vocab_size}")
    print(f"Title vocabulary size: {title_vocab_size}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(content_padded, title_padded, test_size=0.2, random_state=42)

    # Build models
    lstm_model, encoder_model, decoder_model = build_lstm_model(content_vocab_size, title_vocab_size)
    attention_model = build_attention_model(content_vocab_size, title_vocab_size)
    transformer_model = build_transformer_model(content_vocab_size, title_vocab_size)

    # Training configurations
    models = {
        'LSTM': lstm_model,
        'Attention': attention_model,
        'Transformer': transformer_model
    }

    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Create decoder input: all title tokens except the last one
        decoder_input_data = pad_sequences(y_train[:, :-1], maxlen=19, padding='post')
        # Create target data: all title tokens except the first one (start token)
        decoder_target_data = y_train[:, 1:]

        # Check if any index exceeds vocabulary size
        if np.max(decoder_input_data) >= title_vocab_size or np.max(X_train) >= content_vocab_size:
            print("Warning: Some indices exceed vocabulary size!")
            print(f"Max decoder input index: {np.max(decoder_input_data)}, Title vocab size: {title_vocab_size}")
            print(f"Max content index: {np.max(X_train)}, Content vocab size: {content_vocab_size}")

            # Clip indices to be within vocabulary size
            decoder_input_data = np.clip(decoder_input_data, 0, title_vocab_size-1)
            X_train = np.clip(X_train, 0, content_vocab_size-1)

        # Reshape target data for sparse categorical crossentropy
        decoder_target_data = np.expand_dims(decoder_target_data, -1)

        # Train the model


        if name == 'Transformer':
          model.fit(
            [X_train, y_train],  # Input to the encoder and decoder
            y_train,  # Target for the decoder (shifted by one timestep)
            batch_size=16,
            epochs=5,
            validation_split=0.2,
            verbose=1
          )
        else :
          model.fit(
              [X_train, decoder_input_data],
              decoder_target_data,
              batch_size=16,
              epochs=5,
              validation_split=0.2,
              verbose=1
          )

    # Generate and evaluate titles
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name} model...")
        generated_titles = []
        reference_titles = []

        for i in range(min(5, len(X_test))):
            # Clip indices in X_test to be within vocabulary size
            X_test_safe = np.clip(X_test[i], 0, content_vocab_size-1)
            content = ' '.join([content_tokenizer.index_word.get(idx, '') for idx in X_test_safe if idx != 0])

            if name == 'LSTM':
                gen_title = generate_title_lstm(encoder_model, decoder_model, content, content_tokenizer, title_tokenizer, title_vocab_size)
            else:
                gen_title = generate_title(model, content, content_tokenizer, title_tokenizer, content_vocab_size, title_vocab_size)

            # Clip indices in y_test to be within vocabulary size
            y_test_safe = np.clip(y_test[i], 0, title_vocab_size-1)
            ref_title = ' '.join([title_tokenizer.index_word.get(idx, '') for idx in y_test_safe if idx != 0 and title_tokenizer.index_word.get(idx, '') not in ['<start>', '<end>']])

            generated_titles.append(gen_title)
            reference_titles.append(ref_title)

            print(f"Content: {content[:50]}...")
            print(f"Generated: {gen_title}")
            print(f"Reference: {ref_title}")
            print("-" * 50)

        results[name] = evaluate_titles(generated_titles, reference_titles)

    # Print results
    print("\nEvaluation Results:")
    for name, metrics in results.items():
        print(f"{name} Model - Average BLEU: {metrics['avg_bleu']:.4f}")

if __name__ == "__main__":
    main()
