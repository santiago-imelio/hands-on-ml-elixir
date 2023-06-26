defmodule MNIST.Multinomial do
  @moduledoc """
  Multiclass classification over MNIST dataset.
  We predict which number between 0 and 9 a sample is.
  """

  alias Scholar.Linear.LogisticRegression, as: LogReg
  alias Scholar.Metrics

  def train(x, y, num_classes \\ 10) do
    {samples, width, height} = Nx.shape(x)
    x_train = Nx.reshape(x, {samples, width * height})

    LogReg.fit(x_train, y, [num_classes: num_classes, learning_rate: 0.5])
  end

  def predict(model, x) do
    {samples, width, height} = Nx.shape(x)
    x_new = Nx.reshape(x, {samples, width * height})

    LogReg.predict(model, x_new)
  end

  def f1_score(y_true, y_pred, num_classes \\ 10) do
    Metrics.f1_score(y_true, y_pred, [num_classes: num_classes, average: :macro])
  end
end
