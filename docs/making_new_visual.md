# Making a new visual

This page describes the steps for creating a new visual type.

1. Make a new DataStore type. This must be a subclass of the `BaseDataStore` class and implement the `get_slice()` method.
2. Make a new DataStream type. This must be a subclass of the `BaseDataStream` class and implement the `get_data_store_slice()` class.
3. 