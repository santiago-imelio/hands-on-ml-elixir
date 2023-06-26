defmodule MNIST.Binary do
  @moduledoc """
  Binary classification for MNIST dataset.
  We predict if a sample is a 5 or not a 5.
  """

  @sample_dims {28, 28}

  alias Scholar.Linear.LogisticRegression, as: LogReg
  alias Scholar.Metrics

  @doc """
  Map labels to binary classes
  """
  def y_5(y) do
    Nx.map(y, fn label -> Nx.equal(label, 5) end)
  end

  @doc """
  Train a Logistic Regression binary classifier.
  """
  def train_equal_to_five(x, y) do
    {samples, img_width, img_height} = Nx.shape(x)
    x_train = Nx.reshape(x, {samples, img_width * img_height})

    LogReg.fit(x_train, y_5(y), [num_classes: 2, learning_rate: 0.1])
  end

  def predict_equal_to_five(model, %Nx.Tensor{shape: @sample_dims} = sample) do
    {width, height} = Nx.shape(sample)

    x_new =
      sample
      |> Nx.reshape({1, width * height})

    LogReg.predict(model, x_new)
  end

  def predict_equal_to_five(model, samples) do
    {rows, width, height} = Nx.shape(samples)

    x_new =
      samples
      |> Nx.reshape({rows, width * height})

    LogReg.predict(model, x_new)
  end

  def accuracy_score(y_true, y_pred) do
    Metrics.accuracy(y_true, y_pred)
  end

  def confusion_matrix(y_true, y_pred, num_classes \\ 2) do
    Metrics.confusion_matrix(y_true, y_pred, [num_classes: num_classes])
  end

  def precision(y_true, y_pred) do
    Metrics.binary_precision(y_true, y_pred)
  end

  def recall(y_true, y_pred) do
    Metrics.binary_recall(y_true, y_pred)
  end

  def sensitivity(y_true, y_pred) do
    Metrics.binary_sensitivity(y_true, y_pred)
  end

  def f1_score(y_true, y_pred, num_classes \\ 2) do
    Metrics.f1_score(y_true, y_pred, [num_classes: num_classes, average: :macro])
  end
end
