# Making a new visual

This page describes the steps for creating a new visual type.

1. Make a new DataStore type. This must be a subclass of the `BaseDataStore` class and implement the `get_slice()` method.
2. Make a new DataStream type. This must be a subclass of the `BaseDataStream` class and implement the `get_data_store_slice()` class.
3. Make a new visual model (`cellier/models/visuals`)
4. Make a new constructor function that creates the pygfx WorldObject instance from the visual model. These go in `cellier/render`. See `cellier/render/mesh.py` for an example.
5. Add the constructor function to the `construct_pygfx_object()` function in `cellier/render/utils.py`. This function takes the visual model and dispatches the correct constructor function.