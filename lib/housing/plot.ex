defmodule Learning.Housing.Plot do
  alias VegaLite, as: VL

  # Taken from https://hexdocs.pm/scholar/linear_regression.html#california-housing
  def all_features_hist(df) do
    VL.new(
      title: [
        text: "Univariate Histograms of all features"
      ],
      width: 500,
      height: 500,
      columns: 3,
      config: [
        axis: [
          grid: true,
          grid_color: "#dedede"
        ]
      ]
    )
    |> VL.data_from_values(df)
    |> VL.concat(
      for name <- List.delete(df.names, "ocean_proximity") do
        VL.new()
        |> VL.mark(:bar)
        |> VL.encode_field(:x, name, bin: [bin: true, maxbins: 50], axis: [ticks: true])
        |> VL.encode_field(:y, "value count", aggregate: :count)
      end
    )
    |> VL.Viewer.show()
  end

  def income_category_hist(df) do
    VL.new(
      title: [
        text: "Income category histogram"
      ],
      width: 500,
      height: 500,
      columns: 3,
      config: [
        axis: [
          grid: true,
          grid_color: "#dedede"
        ]
      ]
    )
    |> VL.data_from_values(df)
    |> VL.mark(:bar)
    |> VL.encode_field(:x, "income_category", axis: [ticks: true])
    |> VL.encode_field(:y, "value count", aggregate: :count)
    |> VL.Viewer.show()
  end

  def location_scatter_plot(df) do
    VL.new(
      title: [
        text: "Latitude-longitude scatterplot"
      ],
      width: 800,
      height: 650,
      config: [
        axis: [
          grid: true,
          grid_color: "#dedede"
        ]
      ]
    )
    |> VL.data_from_values(df)
    |> VL.mark(:point)
    |> VL.encode_field(
      :x,
      "longitude",
      [
        type: :quantitative,
        scale: [
          zero: false
        ]
      ]
    )
    |> VL.encode_field(
      :y,
      "latitude",
      [
        type: :quantitative,
        scale: [
          zero: false
        ]
      ]
    )
    |> VL.encode(:color, [
      field: "median_house_value",
      type: :quantitative,
      scale: [
        scheme: "rainbow",
      ]
    ])
    |> VL.encode(:size, [
      type: :quantitative,
      field: "population"
    ])
    |> VL.Viewer.show()
  end
end
