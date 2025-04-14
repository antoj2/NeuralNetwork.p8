from collections.abc import Iterable
from itertools import batched
import warnings
import numpy as np
from numpy.typing import NDArray
import csv
import pygame
from PIL import Image


# Inspred by 3blue1brown's series on neural networks
# https://www.3blue1brown.com/topics/neural-networks

np.set_printoptions(suppress=True)
warnings.simplefilter("error")


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
  return 1.0 / (1.0 + np.exp(-x))


# # Derivative of sigmoid function
def sigmoid_prime(x: NDArray[np.float64]) -> NDArray[np.float64]:
  return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
  weights: list[NDArray[np.float64]] = [
    np.zeros((16, 784)),
    np.zeros((16, 16)),
    np.zeros((10, 16)),
  ]
  biases: list[NDArray[np.float64]] = [
    np.zeros(16),
    np.zeros(16),
    np.zeros(10),
  ]
  eta: float = 0
  sizes: list[int] = []

  def __init__(self, eta: float = 10):
    self.sizes = [784, 16, 16, 10]

    self.biases = [np.random.randn(y) * 0.01 for y in self.sizes[1:]]
    self.weights = [
      np.random.randn(y, x) * np.sqrt(1 / x)
      for x, y in zip(self.sizes[:-1], self.sizes[1:])
    ]

    print(self.biases[0].shape)

    self.eta = eta

  def update(self, mini_batch: Iterable[list[str]]):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    rng = np.random.default_rng()
    shuffled = list(mini_batch)
    rng.shuffle(shuffled)

    batch_size = len(shuffled)

    for row in shuffled:
      x = np.array([float(n) / 255 for n in row[1::]], dtype=np.float64)
      y = np.zeros(10, dtype=np.float64)
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

  def backprop(self, x: NDArray[np.float64], y: NDArray[np.float64]):
    nabla_w: list[NDArray[np.float64]] = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    activation = x
    activations = [x]
    zs: list[NDArray[np.float64]] = []

    for w, b in zip(self.weights, self.biases):
      z: NDArray[np.float64] = np.dot(w, activation) + b
      zs.append(z)
      activation: NDArray[np.float64] = sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(np.atleast_2d(delta).T, np.atleast_2d(activations[-2]))

    for layer in range(2, len(self.sizes)):
      z = zs[-layer]
      sp = sigmoid_prime(z)
      delta: NDArray[np.float64] = np.dot(self.weights[-layer + 1].T, delta) * sp
      nabla_b[-layer] = delta
      nabla_w[-layer] = np.dot(
        np.atleast_2d(delta).T, np.atleast_2d(activations[-layer - 1])
      )

    print_results(activations[-1], y)

    return (nabla_w, nabla_b)

  def feedforward(self, a: NDArray[np.float64]):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a

  @staticmethod
  def cost_derivative(
    a: NDArray[np.float64], y: NDArray[np.float64]
  ) -> NDArray[np.float64]:
    return a - y


def main():
  print("Hello from nn!")

  nn = NeuralNetwork()

  with open("mnist_train.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    mini_batches = batched(reader, 100)
    for i, mini_batch in enumerate(mini_batches):
      print("mini batch:", i)
      nn.update(mini_batch)

  with open("mnist_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for test in reader:
      print("\n### TEST ###\n")
      x = np.array([float(n) / 255 for n in test[1::]], dtype=np.float64)
      a = nn.feedforward(x)
      y = np.zeros(10, dtype=np.float64)
      y[int(test[0])] = 1.0
      print_results(a, y)
      print_img(x)

  # User input
  print("#### USER INPUT ###")
  print("Press ENTER to draw a digit, BACKSPACE to clear the screen.")
  _ = pygame.init()
  screen = pygame.display.set_mode((800, 800))  # 28 times 36
  pygame.display.set_caption("Neural Network")
  clock = pygame.time.Clock()
  running = True

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_BACKSPACE:
          _ = screen.fill("black")
        if event.key == pygame.K_RETURN:
          img_array = get_screen_array(screen) / 255
          print_img(img_array)
          a = nn.feedforward(img_array)
          print_results(a)

    button = pygame.mouse.get_pressed()[0]
    if button:
      x, y = pygame.mouse.get_pos()
      _ = pygame.draw.circle(screen, "white", (x, y), 35)

    pygame.display.flip()

    _ = clock.tick(20000)

  pygame.quit()


def get_screen_array(screen: pygame.Surface):
  data = pygame.surfarray.array3d(screen)
  data = np.transpose(data, (1, 0, 2))
  img = Image.fromarray(data)
  img = img.convert("L")
  img = img.resize((28, 28), Image.Resampling.LANCZOS)
  img_array = np.asarray(img).ravel()
  print(img_array.shape)
  return img_array


def print_img(x: NDArray[np.float64]):
  """Expects a 1D array of 784 elements (28x28 pixels) with values between 0 and 1,
  representing a grayscale image."""
  for row in batched(np.atleast_1d(x), 28):
    for pixel in row:
      print(
        f"\x1b[38;2;{int(pixel * 255)};{int(pixel * 255)};{int(pixel * 255)}m██\x1b[0m",
        end="",
      )
    print()


def print_results(x: NDArray[np.float64], y: NDArray[np.float64] | None = None):
  if y is None:
    print(np.column_stack((np.arange(10), np.round(x, 2))))
    return

  print(np.column_stack((np.arange(10), y, np.round(x, 2))))


if __name__ == "__main__":
  main()
