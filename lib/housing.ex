defmodule Learning.Housing do
  alias Explorer.DataFrame, as: DF

  def load_housing_data do
    url = "https://raw.githubusercontent.com/ageron/data/main/housing.tgz"
    {:ok, %Req.Response{status: 200, body: raw_data}} = Req.get(url)

    [{'housing/housing.csv', raw_csv}] = raw_data

    # Load csv into DataFrame
    housing_df = DF.load_csv!(raw_csv)

    # Explore first 5 rows of dataframe
    DF.head(housing_df)
    |> DF.table()
    |> IO.inspect()

    # Show Summary
    DF.head(housing_df)
    |> DF.describe()

    housing_df
  end

  def shuffle_and_split_data(dataframe, test_ratio \\ 0.20) do
    shuffled_data = DF.shuffle(dataframe, seed: 42)
    total_data_size = DF.n_rows(dataframe)
    test_data_size = trunc(total_data_size * test_ratio)
    test_data = DF.head(shuffled_data, test_data_size)
    train_data = DF.tail(shuffled_data, total_data_size - test_data_size)

    %{"train" => train_data, "test" => test_data}
  end
end
