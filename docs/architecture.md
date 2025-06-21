# Cellier Architecture Overview

## Introduction

NOTE: this document is work in progress.

Cellier is a toolkit for building custom data viewers using composable components. Rather than providing a monolithic viewer application, Cellier offers a set of architectural components that developers can combine to create rich, interactive visualizations tailored to their specific needs.

The primary use case is for algorithm and pipeline developers who need to quickly create custom viewers for their data without building everything from scratch.

## Overall Design Philosophy

Cellier follows a **Model-View-Controller (MVC)** architectural pattern with an event-driven communication system:

- **Model**: The `ViewerModel` contains all application state
- **View**: GUI components and the rendering system 
- **Controller**: The `CellierController` coordinates between model and view
- **Events**: An `EventBus` handles all inter-component communication

This separation enables loose coupling between components, making the system highly modular and extensible.

## Core Components

### CellierController
The `CellierController` is the central coordinator that orchestrates the entire system. It:

- Maintains the `ViewerModel` (application state)
- Coordinates between the model and rendering system
- Manages the data slicing pipeline
- Handles user interactions and updates

**Key Responsibilities:**
- Triggering data reslicing when the view changes
- Adding/removing visuals and data stores
- Managing visual callbacks and interactions
- Synchronizing model state with the renderer

### ViewerModel
The `ViewerModel` is a Pydantic dataclass that represents the complete state of the viewer. It contains:

- **DataManager**: Manages all data stores in the system
- **SceneManager**: Manages scenes, each containing visuals and canvases

This separation reflects a key design principle: data and visuals are separate concepts. Multiple visuals can reference the same data store, preventing data duplication and enabling efficient memory usage.

```
ViewerModel
├── DataManager
│   └── stores: Dict[DataStoreId, DataStore]
└── SceneManager
    └── scenes: Dict[SceneId, Scene]
        ├── visuals: List[Visual]
        ├── canvases: Dict[CanvasId, Canvas]
        └── dims: DimsManager
```

### RenderManager
The `RenderManager` handles all rendering operations using PyGFX as the backend. It:

- Creates and manages PyGFX scenes, cameras, and renderers
- Converts Cellier visual models to PyGFX objects
- Handles mouse interactions and camera controls
- Updates visual data when new slices arrive

**Design Note**: All rendering backend code is contained within the `RenderManager`. This encapsulation means other rendering backends could be implemented in the future by creating alternative `RenderManager` implementations. No other backends are planned at this time.

### AsynchronousDataSlicer
The `AsynchronousDataSlicer` manages the threading model for data requests:

- Maintains a thread pool for background data loading
- Manages futures and cancellation for pending requests
- Emits events when data chunks become available
- Handles request prioritization and cleanup

Currently, this is the only asynchronous component, but the architecture allows for additional async components with different scopes in the future.

### EventBus
The `EventBus` implements a signal/subscriber pattern that handles all major inter-component communication. It consists of specialized event buses:

- **VisualEventBus**: Visual model updates and control events
- **SceneEventBus**: Camera, dims, and scene-level events  
- **MouseEventBus**: Mouse interactions and callbacks

Components register their signals with the appropriate bus, and other components subscribe to events they need to respond to. This decouples components and makes the system highly extensible.

## Data Flow: From User Action to Rendering

Here's the typical flow when a user changes a dimension slider:

1. **Reslicing Trigger**: The controller detects the dims change and calls `reslice_visual()`
2. **Region Calculation**: The controller uses the `DimsManager` to determine the current view region
3. **Data Requests**: The `DataStore.get_data_requests()` method creates specific data requests
4. **Async Processing**: Requests are submitted to the `AsynchronousDataSlicer`
5. **Data Loading**: The slicer loads data in background threads
6. **Slice Events**: As data becomes available, the slicer emits `new_slice` events
7. **Rendering Update**: The `RenderManager` receives slice events and updates the visual
8. **Canvas Refresh**: The canvas is redrawn with the new data

## Extension Points

### Data Stores
Data stores handle data access and are the most common extension point. Examples include:
- Lazy-loading stores for large datasets
- Network-based stores for remote data
- Specialized format stores (HDF5, Zarr, etc.)

To create a new data store, implement the `BaseDataStore` interface with:
- `get_data_request()`: Convert view regions into specific data requests
- `get_data()`: Load and return the requested data

### Visuals
Visuals define how data is rendered and displayed. Examples include:
- Points, lines, meshes for geometric data
- Images and volumes for array data
- Custom visualizations for specialized data types

To create a new visual:
1. Create a visual model inheriting from `BaseVisual`
2. Implement the corresponding PyGFX rendering object
3. Add the constructor to the `construct_pygfx_object()` dispatcher

### GUI Frameworks
While Qt is currently supported, the architecture allows for other GUI frameworks by implementing framework-specific canvas widgets and event handlers.

## Key Design Principles

### Separation of Concerns
- **Data vs. Visuals**: Data stores are independent of how data is visualized
- **Model vs. Rendering**: Application state is separate from rendering implementation

### Event-Driven Architecture
- Loose coupling between components
- Easy to add new event types and subscribers
- Clear separation between event producers and consumers

### Composability
- Components can be mixed and matched
- Multiple visuals can share data stores
- Scenes can contain multiple canvases with different views
