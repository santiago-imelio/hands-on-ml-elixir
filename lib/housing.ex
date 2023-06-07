defmodule Learning.Housing do
  require Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Impute.SimpleImputer
  alias Scholar.Preprocessing

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
      |> S.cast(:category)

    DF.put(housing_df, :income_category, income_cat)
  end

  defp bin_median_income(income_value) do
    [0, 1.5, 3.0, 4.5, 6.0, :infinity]
    |> Enum.find_index(&(income_value <= &1))
    |> Integer.to_string()
  end

  def numerical_pipeline(df, attrs) do
    IO.puts("executing numerical pipeline ...")

    x = for name <- attrs, into: [] do
      {name, S.fill_missing(df[name], :nan)}
    end
    |> DF.new()
    |> Nx.stack(axis: 1)

    x
    |> SimpleImputer.fit(strategy: :median)
    |> SimpleImputer.transform(x)
    |> Preprocessing.standard_scale(axes: [0])
  end

  def categorical_pipeline(df, attrs) do
    IO.puts("executing categorical pipeline ...")

    x = for name <- attrs, into: [] do
      {name, S.cast(DF.pull(df, name), :category)}
    end
    |> DF.new()
    |> Nx.stack(axis: 1)

    x
    |> SimpleImputer.fit(strategy: :mode)
    |> SimpleImputer.transform(x)
  end

  def ratio_pipeline(df, attr1, attr2) do
    IO.puts("executing ratio pipeline -> #{attr1} / #{attr2} ...")

    x = for name <- [attr1, attr2], into: [] do
      {name, S.fill_missing(df[name], :nan)}
    end
    |> DF.new
    |> DF.select([attr1, attr2])
    |> Nx.stack(axis: 1)

    x_new =
      x
      |> SimpleImputer.fit(strategy: :median)
      |> SimpleImputer.transform(x)

    col1 = Nx.take(x_new, 0, axis: 1)
    col2 = Nx.take(x_new, 1, axis: 1)

    Nx.divide(col1, col2) |> Nx.reshape({DF.n_rows(df), 1})
  end

  # This takes a while, so not adding it to data pipeline
  def cluster_simil(df, attrs) do
    IO.puts("executing cluster similarity pipeline ...")

    df
    |> DF.select(attrs)
    |> Nx.stack(axis: 1)
    |> ClusterSimilarity.fit_transform()
  end

  def preprocessing(df) do
    df = add_income_category_column(df)

    num_attrs = num_housing_attrs()
    cat_attrs = cat_housing_attrs() ++ ["income_category"]

    Task.await_many([
      Task.async(fn -> ratio_pipeline(df, "total_bedrooms", "total_rooms") end),
      Task.async(fn -> ratio_pipeline(df, "total_rooms", "households") end),
      Task.async(fn -> ratio_pipeline(df, "population", "households") end),
      Task.async(fn -> numerical_pipeline(df, num_attrs) end),
      Task.async(fn -> categorical_pipeline(df, cat_attrs) end),
    ], :infinity)
    |> Nx.concatenate(axis: 1)
  end
end
