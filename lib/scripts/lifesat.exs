alias Explorer.DataFrame, as: DF
alias Explorer.Series, as: S
alias Scholar.Linear.LinearRegression

csv_url = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"

# Download and prepare the data
{:ok, %Req.Response{status: 200, body: raw_csv}} = Req.get(csv_url)

lifesat =
  raw_csv
  |> DF.load_csv!()

# Visualize the data
DF.table(lifesat, limit: :infinity)

%{"GDP per capita (USD)" => x_series} =
  lifesat
  |> DF.select(["GDP per capita (USD)"])
  |> DF.to_series()

%{"Life satisfaction" => y_series} =
  lifesat
  |> DF.select(["Life satisfaction"])
  |> DF.to_series()

# Plot
x_list = S.to_list(x_series)
y_list = S.to_list(y_series)

Enum.zip(x_list, y_list)
|> Contex.Dataset.new(["GDP per capita (USD)", "Life Satisfaction"])
|> Contex.PointPlot.new()
|> Contex.PointPlot.to_svg()

# Prepare data for model
x = S.to_tensor(x_series) |> Nx.reshape({27,1})
y = S.to_tensor(y_series)

# Select a linear model
model = LinearRegression.fit(x, y)

# Make a prediction for Cyprus
x_new = Nx.tensor([[37655.2]]) # Cyprus' GDP per capita in 2020

LinearRegression.predict(model, x_new) |> IO.inspect()
