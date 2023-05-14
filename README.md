# Hands on Machine Learning w/Elixir

**Code examples from the book _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd edition)_ written in Elixir.**

## Example 1-1. Training and running a linear model using Scholar (page 25)

Install the necessary dependencies.
```elixir
def deps do
  [
    {:nx, "~> 0.2"},
    {:scholar, "~> 0.1"},
    {:req, "~> 0.3"},
    {:explorer, "~> 0.5.0"},
    {:vega_lite, "~> 0.1.6"},
  ]
end
```
We'll use [Explorer](https://hexdocs.pm/explorer/Explorer.html) for creating dataframes to explore and manipulate tabular data. For visualizing and creating graphics we'll use [VegaLite](https://hexdocs.pm/vega_lite/VegaLite.html). Finally, [Scholar](https://hexdocs.pm/scholar/Scholar.html) provides some of the traditional ML algorithms built on top of [Nx](https://hexdocs.pm/nx/Nx.html).
```elixir
alias Explorer.DataFrame, as: DF
alias Explorer.Series, as: S
alias Scholar.Linear.LinearRegression
alias VegaLite, as: VL
```
Download and prepare the data
```elixir
{:ok, %Req.Response{status: 200, body: raw_csv}} = Req.get(csv_url)
lifesat = DF.load_csv!(raw_csv)
```
Visualize the data
```elixir
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
|> VL.mark(:point, tooltip: true)
|> VL.encode_field(:x, "GDP per capita (USD)", type: :quantitative, bin: [bin: true, field: "GDP per capita (USD)"])
|> VL.encode_field(:y, "Life satisfaction", type: :quantitative, bin: [bin: true, maxbins: 12, field: "Life satisfaction"])
|> VL.Viewer.show_and_wait()
```
<img width="663" alt="Screen Shot 2023-05-13 at 13 38 06" src="https://github.com/santiago-imelio/hands-on-ml-elixir/assets/82551777/01cec99c-0242-4ec7-b8a3-0932128be98a">

Although data is noisy, it looks like life satisfaction goes up more or less linearly as the country’s GDP per capita increases. Based on this assumption, we decide to model life satisfaction as a linear function of GDP per capita.

```elixir
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
```

With our model we can now make a prediction for a new instance, like Cyprus:
```elixir
x_new = Nx.tensor([[37655.2]]) # Cyprus' GDP per capita in 2020

"Prediction for Cyprus:" |> IO.puts
LinearRegression.predict(model, x_new) |> IO.inspect
```
Output
```
Prediction for Cyprus:
#Nx.Tensor<
  f64[1]
  [6.301657612120332]
>
```

## Figure 2-13. California housing prices (page 63)

The California Housing Prices dataset includes geographical information (latitude and longitude of districts), so it is a good idea to create a scatterplot of all the districts to visualize the data. Start a new IEx session:

```
iex -S mix
```

Run the following to visualize the geographical scatterplot of California Housing dataset:

```
iex(1)> Learning.Housing.load_housing_data |> Learning.Housing.Plot.location_scatter_plot
```

<img width="990" alt="Screen Shot 2023-05-14 at 11 21 29" src="https://github.com/santiago-imelio/hands-on-ml-elixir/assets/82551777/113df858-cf72-4895-8718-2a6371e107e1">

The radius of each circle represents the district’s population, and the color represents the price. This image tells you that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density.

