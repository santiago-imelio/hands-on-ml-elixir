import Config

# Use XLA backend for faster computation
config :nx, :default_backend, {EXLA.Backend, []}
