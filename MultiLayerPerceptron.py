import random
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (작은 랜덤 값으로 설정)
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.hidden_biases = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.output_weights = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.output_bias = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))  # 시그모이드 활성화 함수

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # 시그모이드 미분

    def forward(self, inputs):
        # 은닉층 계산
        self.hidden_outputs = [self.sigmoid(sum(w * x for w, x in zip(weights, inputs)) + b) for weights, b in zip(self.hidden_weights, self.hidden_biases)]
        
        # 출력층 계산
        self.final_output = self.sigmoid(sum(w * h for w, h in zip(self.output_weights, self.hidden_outputs)) + self.output_bias)
        return self.final_output

    def train(self, data, labels, lr=0.1, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(data, labels):
                # 순전파
                hidden_outputs = [self.sigmoid(sum(w * x for w, x in zip(weights, inputs)) + b) for weights, b in zip(self.hidden_weights, self.hidden_biases)]
                final_output = self.sigmoid(sum(w * h for w, h in zip(self.output_weights, hidden_outputs)) + self.output_bias)

                # 오차 계산
                output_error = label - final_output
                total_error += output_error ** 2  # MSE (Mean Squared Error)

                # 출력층 가중치 업데이트
                output_delta = output_error * self.sigmoid_derivative(final_output)
                self.output_weights = [w + lr * output_delta * h for w, h in zip(self.output_weights, hidden_outputs)]
                self.output_bias += lr * output_delta

                # 은닉층 가중치 업데이트
                hidden_deltas = [h * (1 - h) * output_delta * w for h, w in zip(hidden_outputs, self.output_weights)]
                for i in range(len(self.hidden_weights)):
                    for j in range(len(self.hidden_weights[i])):
                        self.hidden_weights[i][j] += lr * hidden_deltas[i] * inputs[j]
                    self.hidden_biases[i] += lr * hidden_deltas[i]

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {total_error:.4f}")

# XOR 게이트 학습 데이터
data = [(0,0), (0,1), (1,0), (1,1)]
labels = [0, 1, 1, 0]  # XOR 정답

mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(data, labels, epochs=10000)

# 학습 결과 출력
for x in data:
    print(f"입력: {x}, 출력: {round(mlp.forward(x))}")
