defmodule DeepHousing do
  @moduledoc """
  Train a sequential neural network with California
  Housing dataset (excluding categorical features).
  """

  alias Explorer.DataFrame, as: DF

  @val_size 6512

  def housing_data() do
    {train, test} =
      Housing.load_data()
      |> DF.discard("ocean_proximity")
      |> Utils.shuffle_and_split_data()

    df_x_train = DF.discard(train, "median_house_value")
    df_x_test = DF.discard(test, "median_house_value")

    y_train =
      train
      |> DF.select("median_house_value")
      |> Nx.stack(axis: 1)

    y_test =
      test
      |> DF.select("median_house_value")
      |> Nx.stack(axis: 1)

    x_train = preprocessing(df_x_train)
    x_test = preprocessing(df_x_test)

    # casting everything to f32 cause Axon don't like f64
    {
      {
        Nx.as_type(x_train, :f32),
        Nx.as_type(y_train, :f32)
      },
      {
        Nx.as_type(x_test, :f32),
        Nx.as_type(y_test, :f32)
      }
    }
  end

  defp preprocessing(df) do
    geo_attrs = ["latitude", "longitude"]
    num_attrs = ["longitude", "latitude", "housing_median_age", "total_rooms",
      "total_bedrooms", "population", "households", "median_income"]
    heavy_tail_attrs = ["total_bedrooms", "total_rooms", "population",
      "households", "median_income"]

    Task.await_many([
      Task.async(Housing, :ratio_pipeline, [df, "total_bedrooms", "total_rooms"]),
      Task.async(Housing, :ratio_pipeline, [df, "total_rooms", "households"]),
      Task.async(Housing, :ratio_pipeline, [df, "population", "households"]),
      Task.async(Housing, :log_pipeline, [df, heavy_tail_attrs]),
      Task.async(Housing, :cluster_simil, [df, geo_attrs]),
      Task.async(Housing, :numerical_pipeline, [df, num_attrs]),
    ], :infinity)
    |> Nx.concatenate(axis: 1)
  end

  def model({_, num_features} = _input_shape) do
    Axon.input("input", shape: {nil, num_features})
    |> Axon.dense(50, activation: :relu)
    |> Axon.dense(50, activation: :relu)
    |> Axon.dense(50, activation: :relu)
    |> Axon.dense(1)
  end

  # Divide train set in train and validation
  def train_val_batches(x, y) do
    {samples, _} = Nx.shape(x)

    train = Stream.zip(
      Nx.to_batched(x[0..samples - @val_size], 32),
      Nx.to_batched(y[0..samples - @val_size], 32)
    )

    val = Stream.zip(
      Nx.to_batched(x[@val_size + 1..samples - 1], 32),
      Nx.to_batched(y[@val_size + 1..samples - 1], 32)
    )

    {train, val}
  end

  def fit(model, train_data, val_data) do
    model
    |> Axon.build(compiler: EXLA, mode: :train)
    |> Axon.Loop.trainer(:mean_squared_error, :adam)
    |> Axon.Loop.validate(model, val_data)
    |> Axon.Loop.metric(:mean_absolute_error, "MAE")
    |> Axon.Loop.run(train_data, %{}, epochs: 50, compiler: EXLA)
  end

  def predict(model, params, x_new) do
    Axon.predict(model, params, x_new)
  end

  @doc """
  Calculates RMSE as performance measure
  """
  def performance(y_true, y_pred) do
    Scholar.Metrics.mean_square_error(y_true, y_pred)
    |> Nx.sqrt()
  end
end
