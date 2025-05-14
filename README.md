# Synthetic 2D Neural Network Classification

This project demonstrates how to build, train, and evaluate a simple neural network classifier on a synthetic 2D dataset. It is fully self-contained, requiring no external data downloads.

## 🚀 Features

- **Synthetic Data Generation**: Generate interleaving half-moon shapes using `scikit-learn`.
- **Model Implementation**: Define a small feed-forward neural network in PyTorch.
- **Training Loop**: Train the model with Binary Cross-Entropy loss and the Adam optimizer.
- **Visualization**: Plot data points and the decision boundary over the feature space.
- **Evaluation**: Compute and display classification accuracy on a held-out test set.

## 🛠 Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - `numpy` (array operations)
  - `scikit-learn` (`make_moons`, train/test split, metrics)
  - `matplotlib` (visualizations)
  - `torch`, `torch.nn`, `torch.optim` (modeling and training)

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/synthetic-2d-classifier.git
   cd synthetic-2d-classifier
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   ```text
   numpy
   scikit-learn
   matplotlib
   torch
   ```

## 🎯 Usage

1. **Generate Data** and split into train/test sets:
   ```python
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split

   X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

2. **Define and Train Model** (example in `train.py`):
   ```bash
   python train.py
   ```

   This will:
   - Initialize a `SimpleNet` model.
   - Train for 200 epochs, logging loss every 50 epochs.
   - Save the trained model to `model.pth`.

3. **Visualize Decision Boundary** (example in `visualize.py`):
   ```bash
   python visualize.py
   ```

4. **Evaluate Accuracy** (included in `train.py` or `evaluate.py`):
   ```bash
   python evaluate.py
   ```

## 📂 Project Structure

```
├── data/                  # (Optional) scripts for generating custom data
├── train.py               # Training loop for the neural network
├── visualize.py           # Plot data points and decision boundary
├── evaluate.py            # Compute accuracy on test set
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔧 Customization & Extensions

- **Hyperparameter Tuning**: Modify learning rate, epochs, and network architecture in `train.py`.
- **Different Synthetic Data**: Swap `make_moons` with `make_circles`, `make_blobs`, or custom distributions.
- **Regularization**: Add dropout layers or weight decay to improve generalization.
- **Deployment**: Wrap the model inference in a simple Flask or Streamlit app for interactive exploration.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or new features.

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

Happy experimenting! 🎉
