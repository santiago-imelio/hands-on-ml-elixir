import Config

# Use XLA backend for faster computation
config :nx, :default_backend, {EXLA.Backend, []}
config :nx, :default_defn_options, [compiler: EXLA, client: :host]
