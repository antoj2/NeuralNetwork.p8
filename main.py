import csv
import warnings
from collections.abc import Iterable
from itertools import batched

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import Array, random

# from numpy.typing import NDArray
from PIL import Image

# Inspred by 3blue1brown's series on neural networks
# https://www.3blue1brown.com/topics/neural-networks

# np.set_printoptions(suppress=True, threshold=np.inf)
warnings.simplefilter("error")


def sigmoid(x: Array) -> Array:
  return 1.0 / (1.0 + jnp.exp(-x))


# Derivative of sigmoid function
def sigmoid_prime(x: Array) -> Array:
  sx = sigmoid(x)
  return sx * (1 - sx)


# While NumPy defines scalar types for each dtype, JAX represents all scalars as a 0-d array.
def quantize(x: float, scale: float) -> Array:
  return jnp.clip(jnp.round(x / scale), -127, 127)


class NeuralNetwork:
  weights: list[Array]
  biases: list[Array]
  eta: float
  sizes: list[int]

  def __init__(self, key: jax.Array, eta: float = 10):
    self.sizes = [784, 10, 14, 10]
    # 784 * 10 + 10 * 14 + 14 * 10 = 8120 weights,
    # 10 + 14 + 10 = 34 biases
    # which makes for 8154 parameters

    # self.biases = [np.random.randn(y) for y in self.sizes[1:]]
    # self.weights = [
    #   np.random.randn(y, x) / np.sqrt(x)
    #   for x, y in zip(self.sizes[:-1], self.sizes[1:])
    # ]
    keys = random.split(key, 2 * (len(self.sizes) - 1))
    self.biases = [
      random.normal(keys[2 * i + 1], (self.sizes[i + 1],))
      for i in range(len(self.sizes) - 1)
    ]
    self.weights = [
      random.normal(keys[2 * i], (self.sizes[i + 1], self.sizes[i]))
      / jnp.sqrt(self.sizes[i])
      for i in range(len(self.sizes) - 1)
    ]

    print(self.biases[0].shape)

    self.eta = eta

  def update(self, mini_batch: Iterable[list[str]], lmbda: float, batch_size: int):
    nabla_w = [jnp.zeros_like(w) for w in self.weights]
    nabla_b = [jnp.zeros_like(b) for b in self.biases]

    for row in mini_batch:
      x = jnp.array([float(n) / 255 for n in row[1:]])
      y = jnp.zeros(10).at[int(row[0])].set(1.0)

      delta_nabla_w, delta_nabla_b = self.backprop(x, y)
      nabla_w = [nw + dnw for dnw, nw in zip(delta_nabla_w, nabla_w)]
      nabla_b = [nb + dnb for dnb, nb in zip(delta_nabla_b, nabla_b)]

    # self.weights = [
    #   (1 - self.eta * (lmbda / 60000)) * w - (self.eta / 10) * nw
    #   for w, nw in zip(self.weights, nabla_w)
    # ]
    self.weights = [
      w - (self.eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)
    ]
    self.biases = [
      b - (self.eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)
    ]

  def backprop(self, x: Array, y: Array):
    nabla_w = [jnp.zeros_like(w) for w in self.weights]
    nabla_b = [jnp.zeros_like(b) for b in self.biases]

    activation = x
    activations = [x]
    zs: list[Array] = []

    for w, b in zip(self.weights, self.biases):
      z = jnp.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = jnp.outer(delta, activations[-2])

    for layer in range(2, len(self.sizes)):
      z = zs[-layer]
      sp = sigmoid_prime(z)
      delta: Array = jnp.dot(self.weights[-layer + 1].T, delta) * sp
      nabla_b[-layer] = delta
      # nabla_w[-layer] = jnp.dot(
      #   jnp.atleast_2d(delta).T, jnp.atleast_2d(activations[-layer - 1])
      # )
      nabla_w[-layer] = jnp.outer(delta, activations[-layer - 1])

    return (nabla_w, nabla_b)

  def feedforward(self, a: Array):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(jnp.dot(w, a) + b)
    return a

  @staticmethod
  def cost_derivative(a: Array, y: Array) -> Array:
    return a - y


def main():
  print("Hello from nn!")

  key = random.key(42)
  nn = NeuralNetwork(key, eta=8)

  epochs = 30
  batch_size = 64

  with open("mnist_train.csv", "r") as csvfile:
    for epoch in range(epochs):
      print("Epoch:", epoch)
      _ = csvfile.seek(0)
      reader = csv.reader(csvfile)
      mini_batches = batched(reader, batch_size)
      for i, mini_batch in enumerate(mini_batches):
        print("mini batch:", i)
        nn.update(mini_batch, lmbda=0.1, batch_size=batch_size)

  accuracy = 0

  print("Training done.")

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

  quantized = jnp.vectorize(quantize)
  new_weights: list[Array] = [
    quantized(layer, scalew).astype(np.int8) for layer in nn.weights
  ]
  new_biases: list[Array] = [
    quantized(layer, scaleb).astype(np.int8) for layer in nn.biases
  ]

  print(nn.weights[0])
  print(nn.biases[0])

  nn.weights = [layer.astype(np.float64) * scalew for layer in new_weights]
  nn.biases = [layer.astype(np.float64) * scaleb for layer in new_biases]

  new_accuracy = 0

  with open("mnist_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for test in reader:
      x = np.array([float(n) / 255 for n in test[1::]], dtype=np.float64)
      a = nn.feedforward(x)
      y = np.zeros(10, dtype=np.float64)
      y[int(test[0])] = 1.0
      new_accuracy += np.argmax(a) == int(test[0])

  print("\n### QUANTIZED TEST RESULTS ###\n")
  print(f"Accuracy: {new_accuracy / 10000 * 100}%")

  with open("untitled.p8", "r") as f:
    data = f.readlines()

  data[6] = f"scalew = {scalew}\n"
  data[7] = f"scaleb = {scaleb}\n"

  flattened = np.concatenate(
    [layer.flatten() for layer in new_weights + new_biases], axis=None
  )
  print("Flattened weights shape:", flattened.shape)

  # print("Flattened weights shape:", flattened)
  with open("untitled.p8", "w") as f:
    _ = f.writelines(data)

  with open("untitled.p8", "a") as f:
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


def print_img(x: Array):
  """Expects a 1D array of 784 elements (28x28 pixels) with values between 0 and 1,
  representing a grayscale image."""
  for row in batched(np.atleast_1d(x), 28):
    for pixel in row:
      print(
        f"\x1b[38;2;{int(pixel * 255)};{int(pixel * 255)};{int(pixel * 255)}m██\x1b[0m",
        end="",
      )
    print()


def print_results(x: Array, y: Array | None = None):
  data = [np.arange(10), np.round(x, 2)]
  if y is not None:
    data.insert(1, y)
  print(np.column_stack(data))


if __name__ == "__main__":
  main()
