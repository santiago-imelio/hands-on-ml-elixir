defmodule MNIST do
  def load_data do
    train_data = Scidata.MNIST.download()
    test_data = Scidata.MNIST.download_test()

    train_data = extract_data(train_data)
    test_data = extract_data(test_data)

    {train_data, test_data}
  end

  defp extract_data({images, labels}) do
    {images_binary, images_type, images_shape} = images
    {labels_binary, labels_type, labels_shape} = labels

    x =
      images_binary
      |> Nx.from_binary(images_type)
      |> Nx.reshape(images_shape)
      |> Nx.squeeze()

    y =
      labels_binary
      |> Nx.from_binary(labels_type)
      |> Nx.reshape(labels_shape)

    {x, y}
  end

  @doc """
  Shows a sample in terminal
  """
  def imshow(image) do
    Nx.to_heatmap(image)
  end
end
