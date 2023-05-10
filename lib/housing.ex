alias Explorer.DataFrame, as: DF

defmodule Learning.Housing do
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
  end
end
