alias Explorer.DataFrame, as: DF
alias Explorer.Series, as: S
alias Scholar.Linear.LinearRegression
alias VegaLite, as: VL

csv_url = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"

# Download and prepare the data
{:ok, %Req.Response{status: 200, body: raw_csv}} = Req.get(csv_url)

lifesat =
  raw_csv
  |> DF.load_csv!()

# Visualize the data
VL.new(
  title: [
    text: "Conuntry GDP per capita - Life Satisfaction"
  ],
  width: 600,
  height: 400,
  config: [
    axis: [
      grid: true,
      grid_color: "#dedede"
    ]
  ]
)
|> VL.data_from_values(lifesat)
|> VL.mark(:point, tooltip: true, grid: true)
|> VL.encode_field(:x, "GDP per capita (USD)", type: :quantitative, bin: [bin: true, field: "GDP per capita (USD)"])
|> VL.encode_field(:y, "Life satisfaction", type: :quantitative, bin: [bin: true, maxbins: 12, field: "Life satisfaction"])
|> VL.Viewer.show_and_wait()

# Prepare data for model
%{"GDP per capita (USD)" => x_series} =
  lifesat
  |> DF.select(["GDP per capita (USD)"])
  |> DF.to_series()

%{"Life satisfaction" => y_series} =
  lifesat
  |> DF.select(["Life satisfaction"])
  |> DF.to_series()

x = S.to_tensor(x_series) |> Nx.reshape({27,1})
y = S.to_tensor(y_series)

# Select a linear model
model = LinearRegression.fit(x, y)

# Make a prediction for Cyprus
x_new = Nx.tensor([[37655.2]]) # Cyprus' GDP per capita in 2020

"Prediction for Cyprus:" |> IO.puts
LinearRegression.predict(model, x_new) |> IO.inspect
