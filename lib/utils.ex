defmodule Learning.Utils do
  alias Explorer.DataFrame, as: DF

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
end
