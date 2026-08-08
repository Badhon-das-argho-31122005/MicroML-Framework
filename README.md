# MicroML-Framework
A lightweight, custom Neural Network framework built from scratch using pure Python to understand the inner workings of libraries like PyTorch.
## About
A lightweight, custom Neural Network framework built from scratch using pure Python. This project is designed to demonstrate the inner workings of deep learning libraries like PyTorch and TensorFlow without relying on heavy external dependencies.

## Motivation
To truly master Machine Learning architectures, one must understand how data flows through layers under the hood. This project focuses on building the foundational building blocks of an ML framework using advanced Object-Oriented Programming (OOP) principles in Python.

## Core Python Concepts Demonstrated
* **Advanced OOP (Inheritance & Polymorphism):** Built a scalable `BaseLayer` class and extended it to create custom layers (e.g., `LinearLayer`, `ReLULayer`). All layers share a polymorphic `forward()` method.
* **Magic Methods:** Implemented dunder methods like `__call__` and `__repr__` within a `Sequential` class, allowing the model to be called intuitively like a function: `model(data)`.
* **Decorators:** Developed a custom `@profiler` decorator to measure the execution time and performance of the training loops.

## Why this stands out?
It showcases the ability to translate complex mathematical and structural ML concepts into clean, modular, and Pythonic code.
