defmodule MNIST.Plot do
  alias Scholar.Metrics
  alias Explorer.DataFrame, as: DF
  alias VegaLite, as: VL

  def confusion_matrix(y_true, y_pred, num_classes \\ 10) do
    conf_matrix = Metrics.confusion_matrix(y_true, y_pred, [num_classes: num_classes])
    conf_list = Nx.to_flat_list(conf_matrix)

    {conf_size, _cols} = Nx.shape(conf_matrix)

    labels = Enum.to_list(0..9)

    cm_df =
      DF.new(
        true_label: List.flatten(List.duplicate(labels, conf_size)),
        predicted_label: List.flatten(for label <- labels, do: List.duplicate(label, conf_size)),
        confusion_val: conf_list
      )

    VL.new(
      title: [
        text: "Confusion Matrix"
      ],
      width: 500,
      height: 500
    )
    |> VL.data_from_values(cm_df)
    |> VL.layers([
      VL.new()
      |> VL.mark(:rect)
      |> VL.encode_field(:x, "predicted_label", type: :ordinal)
      |> VL.encode_field(:y, "true_label", type: :ordinal)
      |> VL.encode_field(:color, "confusion_val", [
        type: :quantitative,
        scale: [scheme: "turbo"]
      ]),
      VL.new()
      |> VL.encode_field(:x, "predicted_label", type: :ordinal)
      |> VL.encode_field(:y, "true_label", type: :ordinal)
      |> VL.mark(:text, [
        font_weight: :bold,
        font_size: 14,
        color: "#FFFF"
      ])
      |> VL.encode_field(:text, "confusion_val")
    ])
    |> VL.Viewer.show()
  end

  def normalized_confusion_matrix(y_true, y_pred, num_classes \\ 10) do
    normalized_conf_matrix =
      y_true
      |> Metrics.confusion_matrix(y_pred, [num_classes: num_classes])
      |> normalize_by_rows()

    conf_list = Nx.to_flat_list(normalized_conf_matrix)

    {conf_size, _cols} = Nx.shape(normalized_conf_matrix)

    labels = Enum.to_list(0..9)

    ncm_df =
      DF.new(
        true_label: List.flatten(List.duplicate(labels, conf_size)),
        predicted_label: List.flatten(for label <- labels, do: List.duplicate(label, conf_size)),
        confusion_ratio: conf_list
      )

    VL.new(
      title: [
        text: "Normalized Confusion Matrix by row"
      ],
      width: 500,
      height: 500
    )
    |> VL.data_from_values(ncm_df)
    |> VL.layers([
      VL.new()
      |> VL.mark(:rect)
      |> VL.encode_field(:x, "predicted_label", type: :ordinal)
      |> VL.encode_field(:y, "true_label", type: :ordinal)
      |> VL.encode_field(:color, "confusion_ratio", [
        type: :quantitative,
        scale: [scheme: "turbo"]
      ]),
      VL.new()
      |> VL.encode_field(:x, "predicted_label", type: :ordinal)
      |> VL.encode_field(:y, "true_label", type: :ordinal)
      |> VL.mark(:text, [
        font_weight: :bold,
        font_size: 12,
        color: "#FFFF"
      ])
      |> VL.encode_field(:text, "confusion_ratio", [
        format: ".2f",
      ])
    ])
    |> VL.Viewer.show()
  end

  defp normalize_by_rows(%Nx.Tensor{shape: {rows, _cols}} = x) do
    for i <- 0..rows - 1 do Nx.divide(x[i], Nx.sum(x[i])) end
    |> Nx.stack(axis: 1)
  end
end
