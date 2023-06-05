defmodule HandsOn do
  alias Learning.Housing
  alias Learning.Utils
  alias Explorer.DataFrame, as: DF
  alias Scholar.Linear.LinearRegression
  alias Scholar.Metrics

  def housing do
    # fetch and load california housing dataset data
    housing_df = Housing.load_housing_data()

    # split and shuffle data into training and test
    {train_data_df, test_data_df} = Utils.shuffle_and_split_data(housing_df)

    IO.puts("Train data size: #{DF.n_rows(train_data_df)}")
    IO.puts("Test data size: #{DF.n_rows(test_data_df)}")

    # labels
    y = Nx.concatenate(housing_df[["median_house_value"]])
    y_train = Nx.concatenate(train_data_df[["median_house_value"]])
    y_test = Nx.concatenate(test_data_df[["median_house_value"]])

    # preprocessed training data
    x_train =
      train_data_df
      |> DF.discard("median_house_value")
      |> Housing.preprocessing()

    # preprocessded test data
    x_test =
      test_data_df
      |> DF.discard("median_house_value")
      |> Housing.preprocessing()

    IO.puts("training model ...")

    # train linear model
    model = LinearRegression.fit(x_train, y_train)

    IO.puts("\nDone.\n")

    # predict on test set
    predictions = LinearRegression.predict(model, x_test)

    # calculate errors
    rmse = Metrics.mean_square_error(y_test, predictions) |> Nx.sqrt()
    mae = Metrics.mean_absolute_error(y_test, predictions)

    IO.puts("target mean: #{Nx.mean(y) |> Nx.to_number}")
    IO.puts("root mean square error: #{Nx.to_number(rmse)}")
    IO.puts("mean absolute error: #{Nx.to_number(mae)}")
  end
end
