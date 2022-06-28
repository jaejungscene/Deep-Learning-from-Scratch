import sys                                                               # 부모 디렉터리로 부터
sys.path.append("/Users/jaejungscene/Projects/Machine_Learning/Deep_Learning_from_scratch") # 시작하기 위해
import pickle

f = open(sys.path[-1]+"/dataset/sample_weight.pkl", "rb")
network = pickle.load(f)
(q,w,e,r,t,y) = network
print(type(network))
print(type(network["W1"]))
print(len(network))
print(q)
print(w)
print(e)
print(r)
print(t)
print(y)
f.close()

print("-"*50)

f = open(sys.path[-1]+"/dataset/mnist.pkl", "rb")
y = pickle.load(f)
(q,w,e,r) = y
print(type(y))
print(len(y))
print(q)
print(w)
print(e)
print(r)
f.close()