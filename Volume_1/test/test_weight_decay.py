import numpy as np

weight_decay = 0
weight_decay_lamda = 0.1
input_size = 784
hidden_size_list = [100, 100, 100, 100, 100, 100]
output_size = 10
hidden_layer_num = len(hidden_size_list)
all_size_list = [input_size] + hidden_size_list + [output_size]
params = {}

print(all_size_list)

for idx in range(1, len(all_size_list)):
   scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU를 사용할 때의 권장 초깃값
   print(scale)
   params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])

for idx in range(1, hidden_layer_num+2):
   W = params['W'+str(idx)]
   weight_decay += 0.5 * weight_decay_lamda * np.sum(W**2)

print('weight decay :',weight_decay)