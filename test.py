import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
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

# 加载Keras原生格式的模型
model = load_model('sentiment_analysis_model_1.keras')
print("Model loaded successfully!")

# 提供用户输入功能
while True:
    user_input = input("Enter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predict_sentiment(user_input, model, vocab_to_int)