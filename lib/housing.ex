defmodule Housing do
  require Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Impute.SimpleImputer
  alias Scholar.Preprocessing
  alias DataTransformer, as: T

  def load_data do
    datasets_path =
      "../hands_on"
      |> Path.expand()
      |> Kernel.<>("/datasets")

    csv_path = datasets_path <> "/housing.csv"

    raw_csv =
      if File.exists?(csv_path) do
        File.read!(csv_path)
      else
        url = "https://raw.githubusercontent.com/ageron/data/main/housing.tgz"

        IO.puts("\nFetching data from #{url}...\n")
        [{'housing/housing.csv', csv_data}] = Req.get!(url).body

        File.mkdir!(datasets_path)
        File.write!(csv_path, csv_data)

        csv_data
      end

    DF.load_csv!(raw_csv)
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

  def log_pipeline(df, attrs) do
    x = for name <- attrs, into: [] do
      {name, S.fill_missing(df[name], :nan)}
    end
    |> DF.new()
    |> Nx.stack(axis: 1)
    |> Nx.log()

    x
    |> SimpleImputer.fit(strategy: :median)
    |> SimpleImputer.transform(x)
    |> Preprocessing.standard_scale(axes: [0])
  end

  def cluster_simil(df, attrs) do
    df
    |> DF.select(attrs)
    |> Nx.stack(axis: 1)
    |> ClusterSimilarity.fit_transform()
  end

  def preprocessing(df) do
    df = add_income_category_column(df)

    cat_attrs = ["ocean_proximity"] ++ ["income_category"]
    geo_attrs = ["latitude", "longitude"]
    num_attrs = ["longitude", "latitude", "housing_median_age", "total_rooms",
      "total_bedrooms", "population", "households", "median_income"]
    heavy_tail_attrs = ["total_bedrooms", "total_rooms", "population",
      "households", "median_income"]

    T.run([
      T.run(&ratio_pipeline/3, [df, "total_bedrooms", "total_rooms"]),
      T.run(&ratio_pipeline/3, [df, "total_rooms", "households"]),
      T.run(&ratio_pipeline/3, [df, "population", "households"]),
      T.run(&log_pipeline/2, [df, heavy_tail_attrs]),
      T.run(&cluster_simil/2, [df, geo_attrs]),
      T.run(&numerical_pipeline/2, [df, num_attrs]),
      T.run(&categorical_pipeline/2, [df, cat_attrs])
    ])
    |> Nx.concatenate(axis: 1)
  end
end
