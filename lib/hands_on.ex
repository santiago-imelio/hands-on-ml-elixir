defmodule HandsOn do
  alias Learning.Housing
  alias Learning.Utils
  alias Explorer.DataFrame
  alias Scholar.Linear.LinearRegression

  def housing do
    raw_data = Housing.load_housing_data()
    {train_data, test_data} = Utils.shuffle_and_split_data(raw_data)
    labels = Housing.labels(train_data)

    IO.puts("Train data size: #{DataFrame.n_rows(train_data)}")
    IO.puts("Test data size: #{DataFrame.n_rows(test_data)}")

    train_data
    |> DataFrame.discard("median_house_value")
    |> Housing.preprocessing()
    |> LinearRegression.fit(labels)
  end
end
