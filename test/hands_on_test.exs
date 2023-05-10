defmodule HandsOnTest do
  use ExUnit.Case
  doctest HandsOn

  test "greets the world" do
    assert HandsOn.hello() == :world
  end
end
