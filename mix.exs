defmodule HandsOn.MixProject do
  use Mix.Project

  def project do
    [
      app: :hands_on,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.2"},
      {:scholar, "~> 0.1"},
      {:req, "~> 0.3"},
      {:csv, "~> 3.0"},
      {:explorer, "~> 0.5.0"},
      {:contex, "~> 0.4.0"}
    ]
  end
end
