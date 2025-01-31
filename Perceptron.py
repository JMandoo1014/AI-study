"""
Perceptron

y = f(WX + b)

X : input data
W : 가중치
b : bias(가중치)
f : 활성화 함수
"""
class Perceptron :
    def __init__(self, input_size) :
        # input_size : 입력 데이터의 크기
        self.weights = [0.0] * input_size # 가중치 초기화
        self.bias = 0.0 # 편향 초기화

    def activation(self, x) :
        return 1 if x >= 0 else 0 # 활성화 함수 => 계단함수
    
    def forward(self, inputs) :
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation(weighted_sum)
    
    def train(self, data, labels, lr=0.1, epochs=10) :
        for epoch in range(epochs) :
            for inputs, label in zip(data, labels) :
                prediction = self.forward(inputs)
                error = label - prediction
                self.weights = [w + lr * error * x for w, x in zip(self.weights, inputs)]
                self.bias += lr * error

# AND 게이트 학습 데이터
data = [(0,0), (0,1), (1,0), (1,1)]
labels = [0, 0, 0, 1] # AND 게이트 정답

perceptron = Perceptron(input_size=2)
perceptron.train(data, labels, epochs=10)

for x in data :
    print(f"입력: {x}, 출력: {perceptron.forward(x)}")