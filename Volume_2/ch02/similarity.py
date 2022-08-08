# coding: utf-8
import sys
sys.path.append('/Users/jaejungscene/Projects/Deep_Learning_from_Scratch/Volume_2')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]  # "you"의 단어 벡터
c1 = C[word_to_id['i']]    # "i"의 단어 벡터
print(c0)
print(c1)
print(cos_similarity(c0, c1))
