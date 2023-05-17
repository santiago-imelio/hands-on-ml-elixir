defmodule Learning.Housing do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Impute.SimpleImputer
  alias Learning.Utils

  # Fetch housing data that we'll use to train our model
  def load_housing_data do
    csv_path =
      "../hands_on"
      |> Path.expand()
      |> Kernel.<>("/datasets/housing.csv")

    raw_csv =
      if File.exists?(csv_path) do
        File.read!(csv_path)
      else
        url = "https://raw.githubusercontent.com/ageron/data/main/housing.tgz"

        IO.puts("Fetching data from #{url}...")
        [{'housing/housing.csv', csv_data}] = Req.get!(url).body

        # Save CSV
        File.write!(csv_path, csv_data)

        csv_data
      end

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

  def data_preprocessing_pipeline(housing_df) do
    clean_data_tensor =
      housing_df
      |> DF.discard("ocean_proximity")
      |> Utils.to_series_list()
      |> Enum.map(&Utils.map_nils_to_nan/1)
      |> Enum.map(&S.to_tensor/1)
      |> Enum.map(&Nx.to_list/1)
      |> Nx.tensor()
      |> Nx.transpose()

    simple_imputer = SimpleImputer.fit(clean_data_tensor, strategy: :median)
    SimpleImputer.transform(simple_imputer, clean_data_tensor)
  end
end
