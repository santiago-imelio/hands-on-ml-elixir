defmodule Learning.Housing do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Impute.SimpleImputer
  alias Scholar.Preprocessing
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
    # |> DF.head(500) Cut data for faster training
  end

  def labels(housing_df) do
    housing_df
    |> DF.pull("median_house_value")
    |> S.to_tensor()
    |> Nx.reshape({DF.n_rows(housing_df), 1})
  end

  @doc """
  Drops non-numerical column "ocean_proximity"
  """
  def num_housing_df(housing_df) do
    housing_df
    |> DF.discard("ocean_proximity")
  end

  @doc """
  List of numerical features names for housing dataset.
  Note that we exclude our the feature to predict.
  """
  def num_housing_attrs do
    ["longitude", "latitude", "housing_median_age", "total_rooms",
      "total_bedrooms", "population", "households", "median_income"]
  end

  @doc """
  List of categorical features names for housing dataset
  """
  def cat_housing_attrs do
    ["ocean_proximity"]
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

  def num_housing(housing_df) do
    housing_df
    |> DF.select(num_housing_attrs())
    |> Utils.to_tensor()
  end

  def cat_housing(housing_df) do
    housing_df
    |> DF.select(cat_housing_attrs())
    |> DF.pull("ocean_proximity")
    |> S.cast(:category)
    |> S.to_tensor()
    |> Nx.reshape({1, DF.n_rows(housing_df)})
    |> Nx.transpose()
  end

  def numerical_pipeline(housing_df) do
    housing_df
    |> num_housing
    |> SimpleImputer.fit(strategy: :median)
    |> SimpleImputer.transform(num_housing(housing_df))
    |> Preprocessing.standard_scale(axes: [1])
    |> Nx.transpose()
  end

  def categorical_pipeline(housing_df) do
    housing_df
    |> cat_housing
    |> SimpleImputer.fit(strategy: :mode)
    |> SimpleImputer.transform(cat_housing(housing_df))
    |> Nx.reshape({DF.n_rows(housing_df)})
    |> Preprocessing.one_hot_encode(num_classes: 5)
  end

  def preprocessing(housing_df) do
    processed_data = [
      numerical_pipeline(housing_df),
      categorical_pipeline(housing_df)
    ]

    Nx.concatenate(processed_data, axis: 1)
  end

  def housing_cat_one_hot(housing_df) do
    housing_df
    |> Utils.one_hot_encode_category("ocean_proximity", 5)
  end

  def housing_prepared(housing_df) do
    {train, _test} =
      Utils.shuffle_and_split_data(housing_df)

    preprocessing(train)
  end
end
