import csv
import warnings
from itertools import batched

import numpy as np
import pandas as pd
import pygame
from numpy.typing import NDArray
from PIL import Image

# Inspred by 3blue1brown's series on neural networks
# https://www.3blue1brown.com/topics/neural-networks

np.set_printoptions(suppress=True, threshold=np.inf)
warnings.simplefilter("error")


def sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
  return 1.0 / (1.0 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_prime(x: NDArray[np.float32]) -> NDArray[np.float32]:
  return sigmoid(x) * (1 - sigmoid(x))


def quantize(x: float, scale: float) -> np.int8:
  return np.clip(round(x / scale), -127, 127)


class NeuralNetwork:
  weights: list[NDArray[np.float32]] = []
  biases: list[NDArray[np.float32]] = []
  eta: float = 0
  sizes: list[int] = []

  def __init__(self, eta: float = 10):
    self.sizes = [784, 10, 14, 10]
    # 784 * 10 + 10 * 14 + 14 * 10 = 8120 weights,
    # 10 + 14 + 10 = 34 biases
    # which makes for 8154 parameters

    self.biases = [np.random.randn(y).astype(np.float32) for y in self.sizes[1:]]
    self.weights = [
      np.random.randn(y, x).astype(np.float32) / np.sqrt(x)
      for x, y in zip(self.sizes[:-1], self.sizes[1:])
    ]

    print(self.biases[0].shape)

    self.eta = eta

  def update(self, mini_batch: NDArray[np.float32], batch_size: int):
    nabla_w = [np.zeros_like(w) for w in self.weights]
    nabla_b = [np.zeros_like(b) for b in self.biases]

    for row in mini_batch:
      x = np.array([n / 255 for n in row[1::]], dtype=np.float32)
      y = np.zeros(10, dtype=np.float32)
      y[int(row[0])] = 1.0

      delta_nabla_w, delta_nabla_b = self.backprop(x, y)
      nabla_w = [nw + dnw for dnw, nw in zip(delta_nabla_w, nabla_w)]
      nabla_b = [nb + dnb for dnb, nb in zip(delta_nabla_b, nabla_b)]

    self.weights = [
      w - (self.eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)
    ]
    self.biases = [
      b - (self.eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)
    ]

  def backprop(self, x: NDArray[np.float32], y: NDArray[np.float32]):
    nabla_w: list[NDArray[np.float32]] = [np.zeros_like(w) for w in self.weights]
    nabla_b = [np.zeros_like(b) for b in self.biases]

    activation = x
    activations = [x]
    zs: list[NDArray[np.float32]] = []

    for w, b in zip(self.weights, self.biases):
      z: NDArray[np.float32] = np.dot(w, activation) + b
      zs.append(z)
      activation: NDArray[np.float32] = sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.outer(delta, activations[-2])

    for layer in range(2, len(self.sizes)):
      z = zs[-layer]
      sp = sigmoid_prime(z)
      delta: NDArray[np.float32] = np.dot(self.weights[-layer + 1].T, delta) * sp
      nabla_b[-layer] = delta
      nabla_w[-layer] = np.outer(delta, activations[-layer - 1])

    # print_results(activations[-1], y)

    return (nabla_w, nabla_b)

  def feedforward(self, a: NDArray[np.float32]):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a

  @staticmethod
  def cost_derivative(
    a: NDArray[np.float32], y: NDArray[np.float32]
  ) -> NDArray[np.float32]:
    return a - y


def main():
  print("Hello from nn!")

  nn = NeuralNetwork(eta=8)

  epochs = 30
  data = np.asarray(
    pd.read_csv("mnist_train.csv", header=None).values, dtype=np.float32
  )
  batch_size = 64

  for epoch in range(epochs):
    print("Epoch:", epoch)
    np.random.shuffle(data)
    mini_batches = batched(data, batch_size)
    for i, mini_batch in enumerate(mini_batches):
      print("mini batch:", i)
      mini_batch = np.asarray(mini_batch, dtype=np.float32)
      nn.update(mini_batch, batch_size)

  accuracy = 0

  print("Training done.")

  test_data = np.asarray(
    pd.read_csv("mnist_test.csv", header=None).values, dtype=np.float32
  )

  for test in test_data:
    print("\n### TEST ###\n")
    x = np.array([float(n) / 255 for n in test[1::]], dtype=np.float32)
    a = nn.feedforward(x)
    y = np.zeros(10, dtype=np.float32)
    y[int(test[0])] = 1.0
    print_results(a, y)
    print_img(x)
    accuracy += np.argmax(a) == int(test[0])

  print("\n### TEST RESULTS ###\n")
  print(f"Accuracy: {accuracy / 10000 * 100}%")

  max_valw = max(layer.max() for layer in nn.weights)
  min_valw = min(layer.min() for layer in nn.weights)
  max_valb = max(layer.max() for layer in nn.biases)
  min_valb = min(layer.min() for layer in nn.biases)

  print(f"Weight max: {max_valw}")
  print(f"Weight min: {min_valw}")
  print(f"Weight mean: {[layer.mean() for layer in nn.weights]}")
  print(f"Bias max: {max_valb}")
  print(f"Bias min: {min_valb}")
  print(f"Bias mean: {[layer.mean() for layer in nn.biases]}")

  amaxw = max(max_valw, abs(min_valw))
  amaxb = max(max_valb, abs(min_valb))

  scalew = (2 * amaxw) / 256
  scaleb = (2 * amaxb) / 256
  print(f"Weight scale: {scalew}")
  print(f"Bias scale: {scaleb}")

  quantized = np.vectorize(quantize)
  new_weights: list[NDArray[np.int8]] = [
    quantized(layer, scalew).astype(np.int8) for layer in nn.weights
  ]
  new_biases: list[NDArray[np.int8]] = [
    quantized(layer, scaleb).astype(np.int8) for layer in nn.biases
  ]

  print(nn.weights[0])
  print(nn.biases[0])

  nn.weights = [layer.astype(np.float32) * scalew for layer in new_weights]
  nn.biases = [layer.astype(np.float32) * scaleb for layer in new_biases]

  new_accuracy = 0

  with open("mnist_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for test in reader:
      x = np.array([float(n) / 255 for n in test[1::]], dtype=np.float32)
      a = nn.feedforward(x)
      y = np.zeros(10, dtype=np.float32)
      y[int(test[0])] = 1.0
      new_accuracy += np.argmax(a) == int(test[0])

  print("\n### QUANTIZED TEST RESULTS ###\n")
  print(f"Accuracy: {new_accuracy / 10000 * 100}%")

  with open("neuralnetwork.p8", "r") as f:
    data = f.readlines()

  data[6] = f"scalew = {scalew}\n"
  data[7] = f"scaleb = {scaleb}\n"

  flattened = np.concatenate(
    [layer.flatten() for layer in new_weights + new_biases], axis=None
  )
  print("Flattened weights shape:", flattened.shape)

  # print("Flattened weights shape:", flattened)
  with open("neuralnetwork.p8", "w") as f:
    _ = f.writelines(data)

  with open("neuralnetwork.p8", "a") as f:
    _ = f.truncate(f.tell() - 16437)
    for i, p in enumerate(flattened):
      if i % 64 == 0:
        _ = f.write("\n")
      _ = f.write(f"{int(np.binary_repr(p, width=8), 2):02x}")

  # just to be sure
  print(f"{flattened[0]}")
  print(f"{flattened[0]}")

  decoded = np.astype(flattened[0], np.int8) * scalew
  print(f"Decoded: {decoded}")

  # User input
  # print("#### USER INPUT ###")
  # print("Press ENTER to draw a digit, BACKSPACE to clear the screen.")
  # _ = pygame.init()
  # screen = pygame.display.set_mode((800, 800))
  # drawn = pygame.Surface((800, 800))  # 28 times 36
  # pygame.display.set_caption("Neural Network")
  # clock = pygame.time.Clock()
  # running = True
  # font = pygame.font.Font(None, 40)
  # guess = None
  #
  # while running:
  #   for event in pygame.event.get():
  #     if event.type == pygame.QUIT:
  #       running = False
  #     if event.type == pygame.KEYDOWN:
  #       if event.key == pygame.K_BACKSPACE:
  #         _ = drawn.fill("black")
  #         guess = None
  #       if event.key == pygame.K_RETURN:
  #         img_array = get_surface_array(drawn) / 255
  #         print_img(img_array)
  #         a = nn.feedforward(img_array)
  #         guess = np.argmax(a)
  #         print_results(a * 100)
  #
  #   button = pygame.mouse.get_pressed()[0]
  #   if button:
  #     x, y = pygame.mouse.get_pos()
  #     _ = pygame.draw.circle(drawn, "white", (x, y), 40)
  #
  #   text = font.render(f"Guess: {guess}", True, "white")
  #   text_pos = text.get_rect(bottom=0, top=drawn.get_height() - 50)
  #
  #   _ = screen.blit(drawn)
  #   _ = screen.blit(text, text_pos)
  #
  #   pygame.display.flip()
  #
  #   _ = clock.tick(60)
  #
  # pygame.quit()


def get_surface_array(screen: pygame.Surface):
  data = pygame.surfarray.array3d(screen)
  data = np.transpose(data, (1, 0, 2))
  img = Image.fromarray(data)
  img = img.convert("L")
  img = img.resize((28, 28), Image.Resampling.LANCZOS)
  img_array = np.asarray(img).ravel()
  return img_array


def print_img(x: NDArray[np.float32]):
  """Expects a 1D array of 784 elements (28x28 pixels) with values between 0 and 1,
  representing a grayscale image."""
  for row in batched(np.atleast_1d(x), 28):
    for pixel in row:
      print(
        f"\x1b[38;2;{int(pixel * 255)};{int(pixel * 255)};{int(pixel * 255)}m██\x1b[0m",
        end="",
      )
    print()


def print_results(x: NDArray[np.float32], y: NDArray[np.float32] | None = None):
  data = [np.arange(10), np.round(x, 2)]
  if y is not None:
    data.insert(1, y)
  print(np.column_stack(data))


if __name__ == "__main__":
  main()
