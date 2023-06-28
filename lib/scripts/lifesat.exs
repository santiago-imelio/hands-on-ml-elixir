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
  width: 700,
  height: 500,
  config: [
    axis: [
      grid: true,
      grid_color: "#dedede"
    ]
  ]
)
|> VL.data_from_values(lifesat)
|> VL.encode_field(:x, "GDP per capita (USD)", [
  type: :quantitative,
  scale: [domain: [25000, 65000]]
])
|> VL.encode_field(:y, "Life satisfaction", [
  type: :quantitative,
  scale: [domain: [5.0, 8.0]]
])
|> VL.encode_field(:text, "Country")
|> VL.layers([
  VL.new() |> VL.mark(:point, opacity: 1, size: 8),
  VL.new() |> VL.mark(:text, [
    align: :left,
    baseline: :bottom,
    x_offset: 5,
    y_offset: -5
  ])
])
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

x = S.to_tensor(x_series) |> Nx.reshape({27, 1})
y = S.to_tensor(y_series)

# Select a linear model
model = LinearRegression.fit(x, y)

# Make a prediction for Cyprus
# Cyprus' GDP per capita in 2020
x_new = Nx.tensor([[37655.2]])

"Prediction for Cyprus:" |> IO.puts()
LinearRegression.predict(model, x_new) |> IO.inspect()
