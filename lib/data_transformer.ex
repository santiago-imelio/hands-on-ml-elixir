defmodule DataTransformer do
  @moduledoc """
  Utility functions to run data transformation pipelines
  concurrently.
  """

  def run(transformation, arguments) do
    Task.async(fn -> apply(transformation, arguments) end)
  end

  def run(pipelines) do
    Task.await_many(pipelines, :infinity)
  end
end
