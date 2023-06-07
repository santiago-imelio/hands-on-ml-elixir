defmodule ClusterSimilarity do
  @moduledoc """
  Custom transformer that uses KMeans clusterer to identify
  the main clusters in the training data, and then
  computes Silhouette Coefficient on each sample to measure
  similarity to its own cluster.
  """

  alias Scholar.Cluster.KMeans
  alias Scholar.Metrics.Clustering

  @doc """
  Locate clusters in the data. How many it searches for is
  controlled by the `n_clusters` hyperparameter. After training,
  the cluster centers are available on `clusters` field.
  """
  def fit(x, n_clusters \\ 10, random_state \\ 42) do
    KMeans.fit(x, num_clusters: n_clusters, seed: random_state)
  end

  @doc """
  Measures how similar each sample is to its own cluster
  computing the Silhouette Coefficient for each sample.
  """
  def transform(x, labels, n_clusters \\ 10) do
    Clustering.silhouette_samples(x, labels, num_clusters: n_clusters)
  end

  def fit_transform(x, n_clusters \\ 10, random_state \\ 42) do
    %KMeans{
      clusters: _cluster_centers,
      labels: labels
    } = fit(x, n_clusters, random_state)

    transform(x, labels, n_clusters) |> Nx.stack(axis: 1)
  end
end
