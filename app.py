# modelo_depressao_com_sarcasmo_comentado.py

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, SpatialDropout1D, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization, Layer, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from transformers import pipeline

# -------------------------
# 1. CONFIGURAÇÕES GLOBAIS
# -------------------------
CSV_PATH    = 'sentiment_tweets.csv'
GLOVE_PATH  = 'glove.6B.100d.txt'
MAX_VOCAB   = 20000
MAX_LEN     = 100
EMB_DIM     = 100
BATCH_SIZE  = 64
EPOCHS      = 10
TEST_SIZE   = 0.2
RANDOM_SEED = 42
TF_LOG_DIR  = './logs_sarc'

# -------------------------
# 2. FUNÇÃO DE PRÉ-PROCESSAMENTO
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# 3. CARREGAR DADOS
# -------------------------
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    'message to examine': 'text',
    'label (depression result)': 'label'
})
df = df.dropna(subset=['text', 'label'])
df['text_clean'] = df['text'].astype(str).map(clean_text)

# -------------------------
# 4. EXTRAÇÃO DE SARCASMO
# -------------------------
sarcasm_detector = pipeline(
    'text-classification',
    model='mrm8488/distilbert-finetuned-sarcasm-classification',
    tokenizer='mrm8488/distilbert-finetuned-sarcasm-classification'
)
sarcasm_scores = []
for txt in df['text_clean']:
    out = sarcasm_detector([txt[:512]])[0]
    prob_score = out['score'] if out['label']=='LABEL_1' else (1 - out['score'])
    sarcasm_scores.append(prob_score)
df['sarcasm_score'] = np.array(sarcasm_scores)

# -------------------------
# 5. TOKENIZAÇÃO E PADDING
# -------------------------
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(df['text_clean'])
seqs = tokenizer.texts_to_sequences(df['text_clean'])
X_seq = tf.keras.preprocessing.sequence.pad_sequences(
    seqs, maxlen=MAX_LEN, padding='post', truncating='post'
)
y = df['label'].astype(int).values
s = df['sarcasm_score'].astype(float).values.reshape(-1, 1)

# -------------------------
# 6. MATRIZ DE EMBEDDINGS
# -------------------------
emb_index = {}
with open(GLOVE_PATH, encoding='utf8') as f:
    for line in f:
        vals = line.split()
        w = vals[0]
        emb_index[w] = np.asarray(vals[1:], dtype='float32')

word_index = tokenizer.word_index
num_words = min(MAX_VOCAB, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMB_DIM), dtype='float32')
for w, i in word_index.items():
    if i < MAX_VOCAB and w in emb_index:
        embedding_matrix[i] = emb_index[w]

# -------------------------
# 7. SPLIT E PESOS
# -------------------------
X_tr_seq, X_te_seq, y_tr, y_te, s_tr, s_te = train_test_split(
    X_seq, y, s,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
mcw = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_tr),
    y=y_tr
)
class_weights = {i: mcw[i] for i in range(len(mcw))}

# -------------------------
# 8. FOCAL LOSS
# -------------------------
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        cross_entropy = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        loss = alpha_factor * tf.pow((1 - p_t), self.gamma) * cross_entropy
        return tf.reduce_mean(loss)

# -------------------------
# 9. CRIAR MODELO
# -------------------------
# Entrada de texto
text_input = Input(shape=(MAX_LEN,), name='text_input')
x = Embedding(
    num_words, EMB_DIM,
    weights=[embedding_matrix],
    input_length=MAX_LEN,
    trainable=False
)(text_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32))(x)
x = BatchNormalization()(x)

# Entrada de sarcasmo
sar_input = Input(shape=(1,), name='sarcasm_input')
y_sar = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(sar_input)

# Combinar
combined = concatenate([x, y_sar])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.2)(z)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[text_input, sar_input], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=FocalLoss(),
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 10. CALLBACKS
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    TensorBoard(log_dir=TF_LOG_DIR)
]

# -------------------------
# 11. TREINAMENTO
# -------------------------
history = model.fit(
    [X_tr_seq, s_tr],
    y_tr,
    validation_data=([X_te_seq, s_te], y_te),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------
# 12. AVALIAÇÃO
# -------------------------
y_pred_prob = model.predict([X_te_seq, s_te])
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_te, (y_pred_prob > t).astype(int).reshape(-1)) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_prob > best_t).astype(int).reshape(-1)

print(f'→ Melhor threshold (F1): {best_t:.2f}')
print(classification_report(y_te, y_pred))
print(confusion_matrix(y_te, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Gerar e salvar imagem da matriz de confusão
cm = confusion_matrix(y_te, y_pred)
labels = ['Normal', 'Depressivo']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('matriz_confusao.png')
plt.show()


# -------------------------
# 13. PREDIÇÃO
# -------------------------
def predict(tweet: str):
    txt = clean_text(tweet)
    seq = tokenizer.texts_to_sequences([txt])
    pad_seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN, padding='post', truncating='post'
    )
    out = sarcasm_detector([txt[:512]])[0]
    sar = out['score'] if out['label']=='LABEL_1' else (1 - out['score'])
    sar = np.array([[sar]], dtype='float32')
    prob_arr = model.predict([pad_seq, sar], batch_size=1)
    prob = float(prob_arr[0][0])
    label = 'Depressivo' if prob >= best_t else 'Normal'
    return label, prob

if __name__ == '__main__':
    exemplo_1 = "Oh great, another Monday morning. So excited;"
    lbl, prb = predict(exemplo_1)
    print(f'Input: {exemplo_1}\nPrevisão: {lbl} (prob={prb:.3f})')

    exemplo_2 = "I feel hopeless and alone."
    lbl, prb = predict(exemplo_2)
    print(f'Input: {exemplo_2}\nPrevisão: {lbl} (prob={prb:.3f})')