defmodule Learning.Housing do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S

  # Fetch housing data that we'll use to train our model
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

  # Randomly partition our data into two, sets: training data set that we'll
  # use to train the model; and test data set that we'll use to test it.
  def shuffle_and_split_data(dataframe, test_ratio \\ 0.20) do
    shuffled_data = DF.shuffle(dataframe, seed: 42)
    total_data_size = DF.n_rows(dataframe)
    test_data_size = trunc(total_data_size * test_ratio)
    test_data = DF.head(shuffled_data, test_data_size)
    train_data = DF.tail(shuffled_data, total_data_size - test_data_size)

    %{"train" => train_data, "test" => test_data}
  end

  def add_income_category_column(housing_df) do
    income_cat =
      housing_df
      |> DF.pull("median_income")
      |> S.transform(&bin_median_income/1)

    DF.put(housing_df, :income_category, income_cat)
  end

  defp bin_median_income(income_value) do
    bins = [0, 1.5, 3.0, 4.5, 6.0, :infinity]

    Enum.find_index(bins, fn bin ->
      income_value <= bin
    end)
  end
end
