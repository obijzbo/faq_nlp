import pandas as pd
from sentence_transformers import SentenceTransformer, util
from configs.config_data import ROOT_PATH, DATA_PATH, TRAIN_DATA, TEST_DATA

train_data_path = f"{ROOT_PATH}/{DATA_PATH}/{TRAIN_DATA}"
test_data_path = f"{ROOT_PATH}/{DATA_PATH}/{TEST_DATA}"

train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

train_list = train_data['Question']
train_list_answer = train_data['Answer']
test_list = test_data['Question']
remove_word_list = ['Albert', 'Einstein']
cleaned_train_list = []
cleaned_test_list = []

for train_str in train_list:
    sentence = " "
    token_list = []
    str_list = train_str.split()
    for token in str_list:
        if token not in remove_word_list:
            token_list.append(token)
    for x in token_list:
        sentence += x+" "
    cleaned_train_list.append(sentence)

for test_str in test_list:
    sentence = " "
    token_list = []
    str_list = test_str.split()
    for token in str_list:
        if token not in remove_word_list:
            token_list.append(token)
    for x in token_list:
        sentence += x+" "
    cleaned_test_list.append(sentence)

train_sentence_embedding = model.encode(cleaned_train_list)
test_sentence_embedding = model.encode(cleaned_test_list)

similarity_df = pd.DataFrame(columns=['QUESTION', 'train', 'ANSWER', 'similarity'])

count = 1
for test_sentence, test_embedding in zip(test_list, test_sentence_embedding):
    temp_df = pd.DataFrame(columns=['QUESTION', 'train', 'ANSWER', 'similarity'])
    for train_sentence, train_embedding, answer in zip(train_list, train_sentence_embedding, train_list_answer):
        similarity = util.pytorch_cos_sim(test_embedding, train_embedding)
        similarity = similarity.item()
        temp_df = temp_df.append({'QUESTION': test_sentence,
                                              'train': train_sentence,
                                              'ANSWER': answer,
                                              'similarity': similarity}, ignore_index=True)
    temp_df = temp_df.sort_values(by=['similarity'], ascending=False)
    temp_df = temp_df.iloc[0]
    similarity_df.append(temp_df)

similarity_df.to_csv(f"{ROOT_PATH}/{DATA_PATH}/similarity.csv")
print(similarity_df)
faq_df = similarity_df[['QUESTION', 'ANSWER']]
print(faq_df)
faq_df.to_csv(f"{ROOT_PATH}/{DATA_PATH}/faq.csv")