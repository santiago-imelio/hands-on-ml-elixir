defmodule Learning.Utils do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Preprocessing

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
    df
    |> DF.names()
    |> Enum.map(fn col -> DF.pull(df, col) end)
  end

  @doc """
  Converts dataframe into a tensor. Expects dataframe that
  contains only numerical values.
  """
  def to_tensor(df) do
    df
    |> to_series_list()
    |> Enum.map(&map_nils_to_nan/1)
    |> Enum.map(&S.to_tensor/1)
    |> Enum.map(&Nx.to_list/1)
    |> Nx.tensor()
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

  @doc """
  Converts a categorical attribute dataframe into a one-hot encoded
  tensor. One-hot encoding allows to create a binary attribute for
  each category of a categorical feature. Basically, it turns a
  categorical column in a sparse matrix with 0s and 1s, where each
  row is a one-hot encoded vector.
  """
  def one_hot_encode_category(df, col_name, n_categories) do
    df
    |> DF.pull(col_name)
    |> S.cast(:category)
    |> S.to_tensor()
    |> Preprocessing.one_hot_encode(num_classes: n_categories)
  end
end
