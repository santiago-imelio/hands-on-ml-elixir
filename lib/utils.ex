defmodule Learning.Utils do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S

  def shuffle_and_split_data(%DF{} = dataframe, test_ratio \\ 0.20) do
    test_data_size =
      dataframe
      |> DF.n_rows()
      |> Kernel.*(test_ratio)
      |> trunc()

    train_data_size =
      dataframe
      |> DF.n_rows()
      |> Kernel.-(test_data_size)

    shuffled_data =
      dataframe
      |> DF.shuffle(seed: 42)

    %{
      "train" => DF.head(shuffled_data, train_data_size),
      "test" => DF.tail(shuffled_data, test_data_size)
    }
  end

  def map_nils_to_nan(%S{} = series) do
    nil_to_nan = fn value ->
      if value == nil do
        :nan
      else
        value
      end
    end

    series
    |> S.transform(nil_to_nan)
  end

  @doc """
  Converts a dataframe into list of series, respecting
  the original ordering of columns.
  """
  def to_series_list(df) do
    column_names = DF.names(df)

    Enum.map(column_names, fn col ->
      df
      |> DF.select(col)
      |> DF.to_series()
      |> Map.get(col)
    end)
  end

  @doc """
  Converts a list of column lists into a map of columns.
  Useful for dataframe input. Assumes that
  the names and the columns have the same index.
  """
  def to_columns_map(column_list, col_names) do
    col_names
    |> Enum.zip_with(column_list, fn name, col ->
      %{"#{name}" => col}
    end)
    |> Map.new(fn map ->
      [[col_name], [col_list]] = [
        Map.keys(map),
        Map.values(map)
      ]

      {col_name, col_list}
    end)
  end
end
