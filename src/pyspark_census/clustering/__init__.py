import os

# Define a list of preferred backends
preferred_backends = ['TkAgg', 'Qt5Agg', 'MacOSX', 'GTK3Agg', 'Agg']

# Iterate over the preferred backends and try to set one
for backend in preferred_backends:
    try:
        import matplotlib
        matplotlib.use(backend)
        print(f"Successfully set matplotlib backend to {backend}")
        break
    except Exception as e:
        print(f"Failed to set matplotlib backend to {backend}: {e}")
else:
    print("None of the preferred backends are available. Falling back to the default backend.")

# Now you can safely import pyplot
import matplotlib.pyplot as plt

# Optionally, you can import other modules or perform other initializations
# import my_package.module1
# import my_package.module2