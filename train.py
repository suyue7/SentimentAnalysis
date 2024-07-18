import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from collections import Counter
from string import punctuation


# 定义加载数据的函数
def loadData():
    with open('reviews.txt', 'r') as f:
        reviews = f.read()
    with open('labels.txt', 'r') as f:
        labels = f.read()
    return reviews, labels


# 调用函数
reviews, labels = loadData()


# 数据预处理函数
def dataPreprocess(reviews_str):
    all_text = ''.join([char for char in reviews_str if char not in punctuation])
    review_list = all_text.split('\n')
    all_text = ' '.join(review_list)
    words = all_text.split()
    return review_list, all_text, words


# 调用函数
reviews, all_text, words = dataPreprocess(reviews)

# 单词编码
word_counter = Counter(words)
sorted_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(sorted_vocab, 1)}

# 将评论转换为整数
reviews_ints = []
for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# 标签编码
labels = labels.split('\n')
labels = np.array([1 if label == 'positive' else 0 for label in labels])

# 去除空的评论
non_zero_idx = [i for i, review in enumerate(reviews_ints) if len(review) != 0]
reviews_ints = [reviews_ints[i] for i in non_zero_idx]
labels = np.array([labels[i] for i in non_zero_idx])

# 设置评论最大长度为200，进行填充和截断
seq_len = 200
features = pad_sequences(reviews_ints, maxlen=seq_len, padding='pre', truncating='pre')

# 拆分训练集、验证集和测试集数据
split_train_ratio = 0.8
features_len = len(features)
train_len = int(features_len * split_train_ratio)

train_x, val_x = features[:train_len], features[train_len:]
train_y, val_y = labels[:train_len], labels[train_len:]

val_x_half_len = int(len(val_x) / 2)
val_x, test_x = val_x[:val_x_half_len], val_x[val_x_half_len:]
val_y, test_y = val_y[:val_x_half_len], val_y[val_x_half_len:]

# 输出数据形状
print("\t\t\tFeature Shapes:")
print(f"Train set: \t\t{train_x.shape}\nValidation set: \t{val_x.shape}\nTest set: \t\t{test_x.shape}")

# 定义超参数
lstm_size = 256
lstm_layers = 2
batch_size = 512
learning_rate = 0.01
embed_size = 300

# 构建模型
model = Sequential()
model.add(Embedding(input_dim=len(vocab_to_int) + 1, output_dim=embed_size, input_length=seq_len))
for _ in range(lstm_layers):
    model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_size)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(train_x, train_y, epochs=8, batch_size=batch_size, validation_data=(val_x, val_y), verbose=1)

# 评估模型
test_loss, test_acc = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
print(f"Test accuracy: {test_acc:.3f}")

# 保存模型为Keras原生格式
model.save('sentiment_analysis_model_2.keras')
print("Model saved successfully!")


# 定义预测函数
def predict_sentiment(text, model, vocab_to_int, seq_len=200):
    # 预处理用户输入的文本
    text = ''.join([char for char in text if char not in punctuation])
    words = text.split()
    text_ints = [vocab_to_int.get(word, 0) for word in words]  # 使用.get()来处理未知单词

    # 填充或截断输入文本
    text_ints = pad_sequences([text_ints], maxlen=seq_len, padding='pre', truncating='pre')

    # 使用模型进行预测
    prediction = model.predict(text_ints)

    # 输出分类结果
    if prediction >= 0.5:
        print("Positive sentiment")
    else:
        print("Negative sentiment")


# 加载Keras原生格式的模型（可选，在重新运行时使用）
model = load_model('sentiment_analysis_model.keras')
print("Model loaded successfully!")

# 提供用户输入功能
while True:
    user_input = input("Enter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predict_sentiment(user_input, model, vocab_to_int)


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.optimizers import Adam
# from collections import Counter
# from string import punctuation
#
#
# # 定义加载数据的函数
# def loadData():
#     with open('reviews.txt', 'r') as f:
#         reviews = f.read()
#     with open('labels.txt', 'r') as f:
#         labels = f.read()
#     return reviews, labels
#
#
# # 调用函数
# reviews, labels = loadData()
#
#
# # 数据预处理函数
# def dataPreprocess(reviews_str):
#     all_text = ''.join([char for char in reviews_str if char not in punctuation])
#     review_list = all_text.split('\n')
#     all_text = ' '.join(review_list)
#     words = all_text.split()
#     return review_list, all_text, words
#
#
# # 调用函数
# reviews, all_text, words = dataPreprocess(reviews)
#
# # 单词编码
# word_counter = Counter(words)
# sorted_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
# vocab_to_int = {word: i for i, word in enumerate(sorted_vocab, 1)}
#
# # 将评论转换为整数
# reviews_ints = []
# for review in reviews:
#     reviews_ints.append([vocab_to_int[word] for word in review.split()])
#
# # 标签编码
# labels = labels.split('\n')
# labels = np.array([1 if label == 'positive' else 0 for label in labels])
#
# # 去除空的评论
# non_zero_idx = [i for i, review in enumerate(reviews_ints) if len(review) != 0]
# reviews_ints = [reviews_ints[i] for i in non_zero_idx]
# labels = np.array([labels[i] for i in non_zero_idx])
#
# # 设置评论最大长度为200，进行填充和截断
# seq_len = 200
# features = pad_sequences(reviews_ints, maxlen=seq_len, padding='pre', truncating='pre')
#
# # 拆分训练集、验证集和测试集数据
# split_train_ratio = 0.8
# features_len = len(features)
# train_len = int(features_len * split_train_ratio)
#
# train_x, val_x = features[:train_len], features[train_len:]
# train_y, val_y = labels[:train_len], labels[train_len:]
#
# val_x_half_len = int(len(val_x) / 2)
# val_x, test_x = val_x[:val_x_half_len], val_x[val_x_half_len:]
# val_y, test_y = val_y[:val_x_half_len], val_y[val_x_half_len:]
#
# # 输出数据形状
# print("\t\t\tFeature Shapes:")
# print(f"Train set: \t\t{train_x.shape}\nValidation set: \t{val_x.shape}\nTest set: \t\t{test_x.shape}")
#
# # 定义超参数
# lstm_size = 256
# lstm_layers = 2
# batch_size = 512
# learning_rate = 0.01
# embed_size = 300
#
# # 构建模型
# model = Sequential()
# model.add(Embedding(input_dim=len(vocab_to_int) + 1, output_dim=embed_size, input_length=seq_len))
# for _ in range(lstm_layers):
#     model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
# model.add(Bidirectional(LSTM(lstm_size)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# # 编译模型
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
#
# # 训练模型
# history = model.fit(train_x, train_y, epochs=20, batch_size=batch_size, validation_data=(val_x, val_y), verbose=1)
#
# # 评估模型
# test_loss, test_acc = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
# print(f"Test accuracy: {test_acc:.3f}")
