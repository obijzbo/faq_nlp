import pandas as pd
from sentence_transformers import SentenceTransformer, util
from configs.config_data import ROOT_PATH, DATA_PATH, TRAIN_DATA, TEST_DATA

train_data_path = f"{ROOT_PATH}/{DATA_PATH}/{TRAIN_DATA}"
test_data_path = f"{ROOT_PATH}/{DATA_PATH}/{TEST_DATA}"

train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

train_list = train_data['Question']
train_list_answer = train_data['Answer']
test_list = test_data['Question']

train_sentence_embedding = model.encode(train_list)
test_sentence_embedding = model.encode(test_list)

similarity_df = pd.DataFrame(columns=['QUESTION', 'train', 'ANSWER', 'similarity'])

for test_sentence, test_embedding in zip(test_list, test_sentence_embedding):
    for train_sentence, train_embedding, answer in zip(train_list, train_sentence_embedding, train_list_answer):
        similarity = util.pytorch_cos_sim(test_embedding, train_embedding)
        similarity = similarity.item()
        similarity_df = similarity_df.append({'QUESTION': test_sentence,
                           'train': train_sentence,
                            'ANSWER': answer,
                           'similarity': similarity}, ignore_index=True)

similarity_df = similarity_df.sort_values(by=['similarity'], ascending=False)

similarity_df = similarity_df[:5]
print(similarity_df)
faq_df = similarity_df[['QUESTION', 'ANSWER']]
print(faq_df)