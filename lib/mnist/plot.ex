defmodule MNIST.Plot do
  alias Scholar.Metrics
  alias Explorer.DataFrame, as: DF
  alias VegaLite, as: VL

  def show_confusion_matrix(y_true, y_pred, num_classes \\ 10) do
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
    |> VL.mark(:rect, align: :center)
    |> VL.encode_field(:x, "predicted_label", type: :ordinal)
    |> VL.encode_field(:y, "true_label", type: :ordinal)
    |> VL.encode_field(:color, "confusion_val", type: :quantitative, scale: [scheme: "turbo"])
    |> VL.Viewer.show()
  end
end
