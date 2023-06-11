defmodule HandsOn do
  alias Explorer.DataFrame, as: DF
  alias Scholar.Linear.LinearRegression
  alias Scholar.Metrics

  def train_and_evaluate_housing_model do
    # fetch and load california housing dataset data
    housing_df = Housing.load_data()

    # split and shuffle data into training and test
    {train_data_df, test_data_df} = Utils.shuffle_and_split_data(housing_df)

    IO.puts("Train data size: #{DF.n_rows(train_data_df)}")
    IO.puts("Test data size: #{DF.n_rows(test_data_df)}")

    # labels
    y = Nx.concatenate(housing_df[["median_house_value"]])
    y_train = Nx.concatenate(train_data_df[["median_house_value"]])
    y_test = Nx.concatenate(test_data_df[["median_house_value"]])

    # preprocess both train and test data
    IO.puts("\nRunning data pipeline ...")

    [train_prepared_df, test_prepared_df] = Task.await_many([
      Task.async(Housing, :preprocessing, [train_data_df]),
      Task.async(Housing, :preprocessing, [test_data_df])
    ], :infinity)

    x_train = train_prepared_df |> Nx.stack(axis: 1)
    x_test = test_prepared_df |> Nx.stack(axis: 1)

    IO.puts("\nPreprocessing done. Training model ...")

    # train linear model
    model = LinearRegression.fit(x_train, y_train)

    IO.puts("\nTraining Done.\n")

    # predict on test set
    predictions = LinearRegression.predict(model, x_test)

    # calculate errors
    rmse = Metrics.mean_square_error(y_test, predictions) |> Nx.sqrt()
    mae = Metrics.mean_absolute_error(y_test, predictions)

    IO.puts(":: performance report ::\n")
    IO.puts("> target mean (reference): #{Nx.mean(y) |> Nx.to_number}")
    IO.puts("> RMSE: #{Nx.to_number(rmse)}")
    IO.puts("> MAE: #{Nx.to_number(mae)}\n")

    model
  end
end
