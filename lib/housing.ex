defmodule Housing do
  require Explorer.DataFrame, as: DF
  alias Explorer.Series, as: S
  alias Scholar.Impute.SimpleImputer
  alias Scholar.Preprocessing

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
    bin_median_income = fn income_value ->
      [0, 1.5, 3.0, 4.5, 6.0, :infinity]
      |> Enum.find_index(&(income_value <= &1))
      |> Integer.to_string()
    end

    income_cat =
      housing_df
      |> DF.pull("median_income")
      |> S.transform(bin_median_income)
      |> S.cast(:category)

    DF.put(housing_df, :income_category, income_cat)
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

    Nx.divide(col1, col2)
    |> Nx.reshape({DF.n_rows(df), 1})
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

    cat_attrs = ["ocean_proximity", "income_category"]
    geo_attrs = ["latitude", "longitude"]
    num_attrs = ["longitude", "latitude", "housing_median_age", "total_rooms",
      "total_bedrooms", "population", "households", "median_income"]
    heavy_tail_attrs = ["total_bedrooms", "total_rooms", "population",
      "households", "median_income"]

    new_names =
      ["bedrooms_ratio"] ++
      ["rooms_per_house"] ++
      ["people_per_house"] ++
      Enum.map(heavy_tail_attrs, &("log_#{&1}")) ++
      ["geo_cluster_similarity"] ++
      Enum.map(num_attrs, &("std_#{&1}")) ++
      cat_attrs

    Task.await_many([
      Task.async(__MODULE__, :ratio_pipeline, [df, "total_bedrooms", "total_rooms"]),
      Task.async(__MODULE__, :ratio_pipeline, [df, "total_rooms", "households"]),
      Task.async(__MODULE__, :ratio_pipeline, [df, "population", "households"]),
      Task.async(__MODULE__, :log_pipeline, [df, heavy_tail_attrs]),
      Task.async(__MODULE__, :cluster_simil, [df, geo_attrs]),
      Task.async(__MODULE__, :numerical_pipeline, [df, num_attrs]),
      Task.async(__MODULE__, :categorical_pipeline, [df, cat_attrs])
    ], :infinity)
    |> Nx.concatenate(axis: 1)
    |> DF.new()
    |> DF.rename(new_names)
  end
end
