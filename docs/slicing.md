# Data Slicing in Cellier
NOTE: this document is work in progress.

This document explains how data slicing works in Cellier, covering the flow from requesting a slice to rendering the data. Slicing is the process of extracting and preparing data for visualization based on the current view state.

## Overview

Cellier's slicing system is designed to efficiently handle large, multi-dimensional datasets by only loading and processing the data needed for the current view. The system operates asynchronously to maintain responsive user interactions while processing data in the background.

## Key Components

- **CellierController**: Orchestrates slicing operations and synchronizes model state with the renderer
- **DimsManager**: Manages the current slice selection state (which dimensions and ranges are being viewed)
- **DataStore**: Stores data and handles slice requests (`get_data_request()`) and data retrieval (`get_data()`)
- **AsynchronousDataSlicer**: Executes slice requests asynchronously using a thread pool
- **RenderManager**: Receives sliced data and updates the visual renderer

## High-Level Slicing Flow

```
User interaction → DimsManager update → CellierController.reslice_visual() 
    → DataStore.get_data_request() → AsynchronousDataSlicer.submit()
    → DataStore.get_data() (in thread) → RenderManager._on_new_slice()
    → Visual update → Canvas redraw
```

### Detailed Steps

1. **Slice Request Initiation**
   ```python
   controller.reslice_visual(scene_id, visual_id, canvas_id)
   ```
   **Location**: `CellierController.reslice_visual()`
   
   This is the entry point for triggering a slice operation. The controller looks up the visual model and scene from the provided IDs to begin the slicing process.

2. **Get Current Selection**
   ```python
   selected_region = world_selected_region_from_dims(scene.dims, visual_model)
   ```
   **Location**: `cellier.slicer.utils.world_selected_region_from_dims()`
   
   This function extracts the current slice selection from the DimsManager and converts it into a SelectedRegion object that describes what portion of the data should be visualized (e.g., which dimensions, index ranges, coordinate space).

3. **Generate Data Requests**
   ```python
   requests = data_store.get_data_request(
       selected_region=selected_region,
       tiling_method=TilingMethod.NONE,
       scene_id=scene.id,
       visual_id=visual_model.id
   )
   ```
   **Location**: `DataStore.get_data_request()` (implemented by specific DataStore subclasses)
   
   The DataStore analyzes the selected region and breaks it down into specific data requests. For simple cases, this returns a single request, but for tiled rendering or complex selections, it may return multiple requests that can be processed independently.

4. **Submit Asynchronous Requests**
   ```python
   slicer.submit(request_list=requests, data_store=data_store)
   ```
   **Location**: `AsynchronousDataSlicer.submit()`
   
   The slicer takes the list of data requests and submits them to a thread pool for background processing. It also handles canceling any pending requests for the same visual to ensure only the most recent slice request is processed.

5. **Process Data (in background thread)**
   ```python
   slice_response = data_store.get_data(request)
   ```
   **Location**: `DataStore.get_data()` (implemented by specific DataStore subclasses)
   
   This runs in a worker thread and performs the actual data loading/processing. The DataStore reads the requested data (from memory, disk, network, etc.) and packages it into a DataResponse object containing the sliced data and metadata.

6. **Update Renderer**
   ```python
   render_manager._on_new_slice(slice_response)
   visual_object.set_slice(slice_data=slice_response)
   ```
   **Location**: `RenderManager._on_new_slice()` and visual-specific `set_slice()` methods
   
   When the background thread completes, the slice response is sent back to the main thread. The RenderManager receives the new data and updates the appropriate visual object in the renderer, which then triggers a canvas redraw to display the new slice.

## For Viewer Builders

### Triggering Slicing

Slicing is typically triggered automatically when:
- The user changes the current slice position via dimension controls
- The view changes (pan, zoom, rotation)
- A new visual is added to the scene

You can manually trigger slicing using:
```python
# Reslice a specific visual
controller.reslice_visual(scene_id, visual_id, canvas_id)

# Reslice entire scene
controller.reslice_scene(scene_id)

# Reslice all scenes
controller.reslice_all()
```

### Handling Slice Events

The slicing system emits events that you can connect to:
```python
# Listen for when new slices are available
controller._slicer.events.new_slice.connect(your_callback)

# Listen for canvas redraw requests
controller._render_manager.events.redraw_canvas.connect(your_callback)
```

### Common Patterns

**Connecting dimension changes to slicing:**
```python
# This is typically done automatically by CellierController
controller.events.scene.subscribe_to_dims(
    dims_id=dims_model.id, 
    callback=controller._on_dims_update
)
```

## For Contributors

### Implementing Custom DataStore Types

To add support for new data sources, implement the `BaseDataStore` interface:

```python
class CustomDataStore(BaseDataStore):
    def get_data_request(self, selected_region, tiling_method, visual_id, scene_id):
        # Convert selected region into specific data requests
        # Return list of DataRequest objects
        pass
    
    def get_data(self, request):
        # Fetch actual data for the request
        # Return DataResponse object
        pass
```

#### `get_data_request()` Method

**Purpose**: Converts a high-level slice selection into one or more specific data requests that can be processed independently.

**Parameters**:
- `selected_region`: A `SelectedRegion` object (either `AxisAlignedSelectedRegion` or `PlaneSelectedRegion`) describing what portion of the data to visualize
- `tiling_method`: A `TilingMethod` enum indicating how to chunk the request (currently `NONE` or `LOGICAL_PIXEL`)
- `visual_id`: The unique identifier of the visual requesting the data
- `scene_id`: The unique identifier of the scene containing the visual

**Actions**: This method analyzes the selected region and determines what actual data needs to be loaded. For simple cases, it returns a single `DataRequest`. For tiled rendering or large datasets, it may break the selection into multiple smaller requests that can be processed in parallel.

**Returns**: A list of `DataRequest` objects (either `AxisAlignedDataRequest` or `PlaneDataRequest`)

#### `get_data()` Method

**Purpose**: Performs the actual data loading and processing for a specific data request.

**Parameters**:
- `request`: A `DataRequest` object specifying exactly what data to load, including coordinate ranges, resolution level, and output format

**Actions**: This method does the heavy lifting of data access - reading from files, databases, network sources, or memory. It applies any necessary transformations (coordinate system conversions, resampling, etc.) and packages the result. This method runs in a background thread, so it must be thread-safe.

**Returns**: A `DataResponse` object containing the actual data array and metadata (visual ID, scene ID, resolution level, etc.)

### Understanding the Threading Model

- **Main Thread**: UI updates, event handling, render management
- **Worker Threads**: Data loading and processing via `AsynchronousDataSlicer`
- **Thread Safety**: DataStore implementations must be thread-safe for the `get_data()` method

## Current Limitations

- **No Coordinate Transformations**: Data is currently assumed to be in the same coordinate space as the view. Transformation support is planned for future releases.
- **Limited Tiling**: While the infrastructure exists, advanced tiling methods are not fully implemented.

## Performance Considerations

- The system cancels pending requests when new ones are submitted for the same visual
- Failed cancellations are tracked to prevent processing stale results
- Thread pool size can be configured when creating the `AsynchronousDataSlicer`

## Error Handling

- Failed slice requests are logged but don't crash the application
- Empty slices are handled gracefully by the visual renderers
- Network/IO errors in DataStore implementations should be caught and handled appropriately