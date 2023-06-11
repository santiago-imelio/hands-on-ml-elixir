defmodule Housing.Plot do
  alias VegaLite, as: VL
  alias Explorer.DataFrame, as: DF

  @doc """
  Taken from https://hexdocs.pm/scholar/linear_regression.html#california-housing
  """
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

  def geo_clusters(housing_df, clusters, kmeans_labels) do
    clusters_df = DF.new(Nx.tensor(clusters))
    # kmeans_labels_df = DF.new(Nx.reshape(kmeans_labels, {20640, 1})) |> IO.inspect()

    df_with_kmeans_labels =
      housing_df
      |> DF.put("kmeans_labels", Nx.stack(kmeans_labels, axis: 1))

    VL.new(
      title: [
        text: "Geographic clusters found by KMeans algorithm"
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
    |> VL.layers([
      VL.new(
        width: 800,
        height: 650
      )
      |> VL.data_from_values(df_with_kmeans_labels)
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
      |> VL.encode_field(
        :color,
        "kmeans_labels",
        [
          type: :ordinal,
          scale: [
            scheme: "rainbow"
          ]
        ]
      ),
      VL.new(
        width: 800,
        height: 650
      )
      |> VL.data_from_values(clusters_df)
      |> VL.mark(:point, [
        size: 70,
        color: "#0d0154",
        stroke_width: 10,
        opacity: 1
      ])
      |> VL.encode_field(
        :x,
        "x2",
        [
          type: :quantitative,
          scale: [
            zero: false
          ]
        ]
      )
      |> VL.encode_field(
        :y,
        "x1",
        [
          type: :quantitative,
          scale: [
            zero: false
          ]
        ]
      )
    ])
    |> VL.Viewer.show()
  end

  def geo_similarity(housing_df, clusters, kmeans_labels, silhoutte_samples) do
    clusters_df = DF.new(Nx.tensor(clusters))
    # kmeans_labels_df = DF.new(Nx.reshape(kmeans_labels, {20640, 1})) |> IO.inspect()

    df_with_kmeans_labels =
      housing_df
      |> DF.put("kmeans_labels", Nx.stack(kmeans_labels, axis: 1))
      |> DF.put("cluster_similarity", silhoutte_samples)
      |> IO.inspect()

    VL.new(
      title: [
        text: "Silhouette Coefficient Similarity to each cluster"
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
    |> VL.layers([
      VL.new(
        width: 800,
        height: 650
      )
      |> VL.data_from_values(df_with_kmeans_labels)
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
      |> VL.encode_field(
        :color,
        "cluster_similarity",
        [
          type: :quantitative,
          scale: [
            scheme: "turbo"
          ]
        ]
      ),
      VL.new(
        width: 800,
        height: 650
      )
      |> VL.data_from_values(clusters_df)
      |> VL.mark(:point, [
        size: 70,
        color: "#0d0154",
        stroke_width: 10,
        opacity: 1
      ])
      |> VL.encode_field(
        :x,
        "x2",
        [
          type: :quantitative,
          scale: [
            zero: false
          ]
        ]
      )
      |> VL.encode_field(
        :y,
        "x1",
        [
          type: :quantitative,
          scale: [
            zero: false
          ]
        ]
      )
    ])
    |> VL.Viewer.show()
  end
end
