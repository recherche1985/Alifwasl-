import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings("ignore")

# --- Diacritic Metrics ---
def compute_WER(preds, trues):
    """Calculate Word Error Rate (character-level)"""
    total_chars = sum(len(t) for t in trues)
    wrong_chars = sum(sum(p != t for p, t in zip(pred, true)) 
                     for pred, true in zip(preds, trues))
    return wrong_chars / total_chars if total_chars > 0 else 0.0

def compute_SER(preds, trues):
    """Calculate Sentence Error Rate (fully correct sentences)"""
    correct_sentences = sum(np.array_equal(p, t) for p, t in zip(preds, trues))
    total_sentences = len(preds)
    return 1 - (correct_sentences / total_sentences) if total_sentences > 0 else 0.0

# --- Text Normalization ---
def normalize_text(text, preserve_alif_wasl=False):
    if not preserve_alif_wasl:
        text = text.replace('ٱ', 'ا')
    text = re.sub(r'[إأآ]', 'ا', text)
    return text

# --- Load Data ---
def load_tashkeela(path, limit=None, preserve_alif_wasl=False):
    texts, labels = [], []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
        for line in tqdm(lines, desc="Loading data"):
            line = normalize_text(line.strip(), preserve_alif_wasl)
            if not line: continue
            chars, diacs = [], []
            for ch in line:
                if ch in 'ًٌٍَُِّْ':
                    if chars: diacs[-1] += ch  # Append to last diacritic
                else:
                    chars.append(ch)
                    diacs.append('')
            texts.append(chars)
            labels.append(diacs)
    return texts, labels

def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for seq in data:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def build_label_map(labels_data):
    label_map = {'<PAD>': 0, '<UNK>': 1, '': 2}  # Empty string for no diacritic
    for seq in labels_data:
        for tag in seq:
            if tag not in label_map:
                label_map[tag] = len(label_map)
    return label_map

def encode_and_pad(data, vocab, max_len):
    return pad_sequences(
        [[vocab.get(ch, vocab['<UNK>']) for ch in seq] for seq in data],
        maxlen=max_len, 
        padding='post', 
        value=vocab['<PAD>']
    )

# --- Model Definition ---
def build_model(vocab_size, num_labels, max_len):
    input_layer = Input(shape=(max_len,))
    emb = Embedding(vocab_size, 128, mask_zero=True)(input_layer)
    bi = Bidirectional(LSTM(128, return_sequences=True))(emb)
    output_layer = TimeDistributed(Dense(num_labels, activation='softmax'))(bi)
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def remove_padding(sequences, labels, pad_value=0):
    """Remove padding while maintaining alignment between sequences and labels"""
    cleaned = []
    for seq, label in zip(sequences, labels):
        mask = seq != pad_value
        cleaned.append((seq[mask], label[mask]))
    return cleaned

# --- Training Pipeline ---
def train_and_evaluate(data_path, preserve_alif_wasl, model_name, max_len=150):
    print(f"\n🚀 Training {model_name}...")
    
    # Load and preprocess data
    X_raw, y_raw = load_tashkeela(data_path, preserve_alif_wasl=preserve_alif_wasl)
    char_vocab = build_vocab(X_raw)
    label_map = build_label_map(y_raw)
    
    # Encode and pad sequences
    X = encode_and_pad(X_raw, char_vocab, max_len)
    y = encode_and_pad(y_raw, label_map, max_len)
    
    # Split data (80% train, 10% validation, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Build and train model
    model = build_model(len(char_vocab), len(label_map), max_len)
    history = model.fit(X_train, y_train, 
                       batch_size=32, 
                       epochs=50,
                       validation_data=(X_val, y_val), 
                       verbose=1)
    
    # Prepare cleaned data for metrics
    val_cleaned = remove_padding(X_val, y_val)
    X_val_clean = [x for x, y in val_cleaned]
    y_val_clean = [y for x, y in val_cleaned]
    
    # Get predictions and clean them
    y_pred = np.argmax(model.predict(X_val), axis=-1)
    pred_cleaned = remove_padding(X_val, y_pred)
    y_pred_clean = [y for x, y in pred_cleaned]
    
    # Calculate metrics
    val_acc = history.history['val_accuracy'][-1]
    val_wer = compute_WER(y_pred_clean, y_val_clean)
    val_ser = compute_SER(y_pred_clean, y_val_clean)
    
    print(f"\n📊 {model_name} Validation Results:")
    print(f"✅ Val_Acc: {val_acc*100:.2f}%")
    print(f"🔤 Val_WER: {val_wer:.4f}")
    print(f"📄 Val_SER: {val_ser:.4f}")
    
    return val_acc, val_wer, val_ser

# --- Main Execution ---
if __name__ == "__main__":
    DATA_PATH = "D:/alifwasl/Diacritization/Tashkeela/textstxt/tashkeela_sample.txt"
    MAX_LEN = 150

    # Train and evaluate models
    val_acc_base, val_wer_base, val_ser_base = train_and_evaluate(
        DATA_PATH, preserve_alif_wasl=False, model_name="Baseline", max_len=MAX_LEN
    )
    
    val_acc_alif, val_wer_alif, val_ser_alif = train_and_evaluate(
        DATA_PATH, preserve_alif_wasl=True, model_name="ٱ-aware Model", max_len=MAX_LEN
    )

    # Print comparison table
    print("\n\n📊 Final Comparison Table:")
    print(f"{'Model':<20} {'Val_Acc':<10} {'Val_WER':<10} {'Val_SER':<10}")
    print(f"{'Baseline':<20} {val_acc_base*100:.2f}%   {val_wer_base*100:.2f}%     {val_ser_base*100:.2f}%")
    print(f"{'ٱ-aware Model':<20} {val_acc_alif*100:.2f}%   {val_wer_alif*100:.2f}%      {val_ser_alif*100:.2f}%")
