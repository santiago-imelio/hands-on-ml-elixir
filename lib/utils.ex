defmodule Utils do
  alias Explorer.DataFrame, as: DF

  def shuffle_and_split_data(%DF{} = dataframe, num_rows \\ 0, test_ratio \\ 0.20) do
    reduce_df = fn df ->
      if num_rows == nil || num_rows == 0 do
        df
      else
        DF.head(df, num_rows)
      end
    end

    shuffled_data =
      dataframe
      |> DF.shuffle(seed: 42)
      |> reduce_df.()

    test_data_size =
      shuffled_data
      |> DF.n_rows()
      |> Kernel.*(test_ratio)
      |> trunc()

    train_data_size =
      shuffled_data
      |> DF.n_rows()
      |> Kernel.-(test_data_size)

    {DF.head(shuffled_data, train_data_size),
      DF.tail(shuffled_data, test_data_size)}
  end
end
