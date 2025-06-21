# Chunked 3D Image Rendering System Design Document

## 1. Overview

This design document outlines a chunked 3D image rendering system for Cellier that enables visualization of large volumetric datasets that exceed GPU memory limits. The system implements level-of-detail rendering using pre-computed multiscale image pyramids and frustum culling to load only visible chunks at appropriate resolutions.

The approach prioritizes responsiveness through hierarchical loading: coarse resolution data loads immediately to provide visual feedback, while finer resolution data loads progressively. The system integrates with Cellier's existing asynchronous data pipeline and event-driven architecture, providing a new `ChunkedImageStore` that implements the standard `BaseDataStore` interface while maintaining clean separation between data models and rendering.

Key technical strategies include:
- **Frustum culling** to determine visible chunks based on camera perspective
- **Level-of-detail selection** based on camera distance and screen-space chunk size
- **Hierarchical loading** with coarse-to-fine progression for responsiveness
- **Separated GPU texture management** within the rendering layer
- **Request cancellation** for responsive camera interactions

## 2. Goals

### Primary Goals
- **Memory Efficiency**: Render 3D volumes larger than available GPU memory by loading only visible chunks
- **Adaptive Quality**: Automatically select appropriate resolution levels based on camera state (distance, zoom level)
- **Responsive Interaction**: Maintain smooth camera movement by showing lower quality data during interaction and upgrading quality when idle
- **Progressive Loading**: Always show something immediately, then progressively enhance quality
- **Multi-format Support**: Work with chunked array formats (Zarr, HDF5) over both local and network storage
- **Integration**: Seamlessly integrate with existing Cellier architecture (DataStore interface, async pipeline, event system)

### Secondary Goals  
- **Clean separation of concerns**: Maintain separation between data models and rendering implementation
- **Extensible Texture Management**: Design texture allocation system that can evolve from simple to sophisticated approaches
- **Configurable Performance**: Allow memory budgets and loading strategies to be tuned for different hardware
- **Observable Loading**: Provide visual feedback about data loading progress and completeness
- **Error Resilience**: Handle network failures and data corruption gracefully without crashing

### Non-Goals
- **Data Writing**: Initial implementation focuses on read-only visualization (though architecture supports future writing)
- **Cross-scale Interpolation**: No blending between resolution levels (discrete quality jumps are acceptable)
- **Automatic Pyramid Generation**: Users must provide pre-computed multiscale pyramids

## 3. Architecture

The chunked rendering system consists of components split between the data layer and rendering layer to maintain Cellier's clean separation of concerns:

```
Data Layer (Models)                    Rendering Layer (PyGFX)
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        CellierController        │    │         RenderManager           │
│       reslice_visual()          │    │        _on_new_slice()          │
└────────────┬────────────────────┘    └─────────────┬───────────────────┘
             │                                       │
             ▼                                       │
┌─────────────────────────────────┐                  │
│      ChunkedImageStore          │                  │
│  get_data_request() / get_data()│                  │
└────────────┬────────────────────┘                  │
             │                                       │
   ┌─────────┼─────────┐                            │
   ▼         ▼         ▼                            │
┌─────────┐ ┌─────────┐ ┌─────────┐                  │
│LevelOf  │ │ Chunk   │ │ Chunk   │                  │
│Detail   │ │ Culler  │ │ Manager │──numpy arrays────┤
│Selector │ │         │ │         │                  │
└─────────┘ └─────────┘ └─────────┘                  ▼
                        │                   ┌─────────────────┐
                        │                   │   TextureAtlas  │
                        ▼                   │ (PyGFX Textures)│
             ┌─────────────────────────┐    └─────────────────┘
             │ AsynchronousDataSlicer  │
             │        submit()         │
             └─────────────────────────┘
```

**Component Responsibilities:**

**Data Layer:**
- **ChunkedImageStore**: Main entry point, implements `BaseDataStore` interface, returns numpy arrays
- **LevelOfDetailSelector**: Determines which resolution scales to load based on camera state
- **ChunkCuller**: Performs frustum culling to find visible chunks within each scale
- **ChunkManager**: Handles CPU-side chunk caching, loading prioritization, and cancellation
- **LoadingProgressTracker**: Provides status callbacks to visuals for loading feedback

**Rendering Layer:**
- **TextureAtlas** (inside RenderManager): Manages GPU memory allocation and PyGFX texture objects
- **RenderManager**: Coordinates GPU upload and visual updates

**Data Flow:**
1. User interaction triggers `CellierController.reslice_visual()`
2. `ChunkedImageStore.get_data_request()` uses `LevelOfDetailSelector` and `ChunkCuller` to determine required chunks
3. `ChunkManager` checks CPU cache, prioritizes missing chunks, and submits I/O requests to `AsynchronousDataSlicer`
4. As chunks load into CPU memory, `ChunkManager` emits events with numpy arrays
5. `RenderManager` receives numpy arrays, uses `TextureAtlas` to upload to GPU textures
6. `RenderManager` updates visuals with new texture coordinates and triggers canvas refresh

## 4. Implementation

### 4.1 ChunkedImageStore

**Design**: The main DataStore implementation that orchestrates the chunked loading system. Maintains multiscale metadata and delegates chunk management to specialized components. Returns numpy arrays only - no GPU-specific data.

**Key Considerations**:
- Must implement existing `BaseDataStore` interface for seamless integration
- Handles both axis-aligned and plane-based region selections
- Manages scale metadata (chunk shapes, downscale factors, data ranges)
- Coordinates between level-of-detail selection and chunk culling
- Returns CPU data only, maintaining separation from rendering concerns

```python
class ChunkedImageStore(BaseDataStore):
    def __init__(self, scales, chunk_shapes, downscale_factors):
        # Initialize multiscale data structures (no GPU memory budget here)
        # Create subsystem components (LevelOfDetailSelector, ChunkCuller, ChunkManager)
        # Set up progress tracking for loading status
    
    def get_data_request(self, selected_region, tiling_method, visual_id, scene_id):
        # Extract frustum corners from selected region
        # Use LevelOfDetailSelector to determine target scales (coarse to fine)
        # For each scale, use ChunkCuller to find visible chunks
        # Calculate priority for each chunk based on distance from camera
        # Return list of ChunkedDataRequest objects ordered by priority
    
    def get_data(self, request):
        # Delegate to ChunkManager to handle actual data loading
        # Return ChunkedDataResponse with numpy arrays and metadata
        # NO texture coordinates or GPU-specific information
```

### 4.2 LevelOfDetailSelector

**Design**: Determines which resolution scales to load based on camera state and viewing parameters. Implements a distance-based heuristic that can be easily swapped for more sophisticated algorithms.

**Key Considerations**:
- Must balance quality vs performance by selecting appropriate scales
- Should return multiple scales for hierarchical loading (coarse fallback + fine target)
- Algorithm should be easily replaceable for future enhancements
- Consider both camera distance and screen-space chunk size

```python
class LevelOfDetailSelector:
    def __init__(self, scales, downscale_factors):
        # Store scale metadata for decision making
    
    def select_scales(self, frustum_corners, region, max_scales=3):
        # Calculate characteristic viewing distance from frustum
        # Estimate screen-space size of chunks at each scale
        # Determine target scale based on distance heuristics
        # Build hierarchical scale list: [coarsest_for_immediate_feedback, intermediate, target]
        # Return ordered list from coarsest to finest (max 3 scales)
    
    def _calculate_view_distance(self, frustum_corners):
        # Compute distance from camera to center of near frustum plane
        # This drives scale selection - closer viewing needs higher resolution
    
    def _select_target_scale(self, view_distance):
        # Use distance thresholds to select appropriate target scale
        # Configurable thresholds allow tuning for different datasets
```

### 4.3 ChunkCuller

**Design**: Performs frustum culling to determine which chunks intersect the viewing volume. Uses efficient AABB-frustum intersection tests from existing geometry utilities.

**Key Considerations**:
- Must handle chunks of different sizes across scales
- Should leverage existing frustum intersection code
- Performance critical - will be called frequently during camera movement
- Must account for chunk coordinate systems and transformations

```python
class ChunkCuller:
    def find_visible_chunks(self, frustum_corners, scale, culling_mode="any"):
        # Convert frustum corners to plane equations for intersection testing
        # Iterate through all chunks in the scale level
        # For each chunk, test its 8 corner points against frustum planes
        # Apply culling mode: "any" (any corner inside) or "all" (all corners inside)
        # Return list of ChunkId objects for visible chunks
        # Use existing geometry utilities for efficient intersection tests
```

### 4.4 ChunkManager

**Design**: Central coordinator for chunk lifecycle management in CPU memory. Handles caching, loading prioritization, and request cancellation. Focuses purely on data management without GPU concerns.

**Key Considerations**:
- Must integrate with existing `AsynchronousDataSlicer` for I/O operations
- Implements request cancellation when views change rapidly
- Manages chunk priority queue based on distance from camera
- Maintains CPU-side cache of chunk data (numpy arrays)
- Provides loading status callbacks for visual feedback
- Emits events with numpy arrays for RenderManager to handle GPU upload

```python
class ChunkManager:
    def __init__(self, scales, slicer):
        # Initialize CPU chunk cache and loading state tracking
        # Set up priority queue for chunk requests
        # Create progress tracker for status callbacks
        # NO GPU memory management - that's handled by RenderManager
    
    def request_chunks(self, chunk_requests, visual_id):
        # Cancel any previous requests for this visual to ensure responsiveness
        # Separate chunks into available (CPU cached) vs missing (need loading)
        # For missing chunks, add to priority queue based on distance from camera
        # Submit high-priority chunks to AsynchronousDataSlicer immediately
        # Return ChunkedDataResponse with available numpy arrays and loading status
    
    def _schedule_chunk_loading(self, request):
        # Add chunk to priority queue if not already loading
        # Priority based on distance from front clipping plane along view ray
        # Submit to background thread pool via AsynchronousDataSlicer
    
    def _process_loading_queue(self):
        # Process priority queue of pending chunk loads
        # Skip cancelled requests (visual no longer active)
        # Submit chunk I/O tasks to thread pool with completion callbacks
    
    def _load_chunk_data(self, chunk_id):
        # Background thread function to load chunk from storage (Tensorstore, zarr, HDF5)
        # Handle network/local file systems via storage libraries
        # Log warnings for failures but don't crash program
        # Return numpy array of chunk data
    
    def _on_chunk_loaded(self, future, chunk_id):
        # Handle completion of background chunk loading
        # Cache loaded numpy data in CPU memory
        # Emit event with numpy array for RenderManager to handle GPU upload
        # Emit progress callbacks to registered visuals
        # Handle errors gracefully with logging
    
    def _cancel_visual_requests(self, visual_id):
        # Cancel futures for all chunks belonging to this visual
        # Remove from active request tracking
        # Ensures responsive camera movement by avoiding stale loads
```

### 4.5 TextureAtlas (Inside RenderManager)

**Design**: Manages GPU texture allocation and coordinate mapping for chunks within the rendering layer. Handles PyGFX-specific texture objects and GPU memory management. Phase 1 implementation uses simple pre-allocated textures per scale level.

**Key Considerations**:
- Lives inside RenderManager to maintain model/renderer separation
- Manages PyGFX Texture objects directly
- Fixed memory budget per scale level for predictable performance
- Must handle different chunk sizes across scales
- Provides texture coordinate mapping for shader access
- Interface designed for easy transition to more sophisticated allocation strategies
- LRU eviction when GPU memory is full

```python
class TextureAtlas:  # Inside RenderManager
    def __init__(self, max_gpu_memory_bytes):
        # Allocate GPU memory budget across scale levels
        # Initialize PyGFX Texture storage and region tracking
        # Set up LRU eviction system for GPU memory management
    
    def upload_chunk(self, chunk_id, numpy_data):
        # Ensure PyGFX Texture exists for chunk's scale level
        # Allocate space within scale texture using grid-based allocator
        # Handle eviction if texture is full (LRU policy)
        # Upload numpy data to PyGFX Texture and return TextureRegion with coordinates
        # Update access tracking for LRU management
    
    def get_chunk_region(self, chunk_id):
        # Return TextureRegion for chunk if loaded in GPU memory
        # Update LRU access tracking
        # Return None if chunk not currently in GPU memory
    
    def _allocate_scale_texture(self, scale_level, chunk_shape):
        # Create PyGFX 3D Texture for scale level based on memory budget
        # Calculate texture dimensions to fit maximum expected chunks
        # Use simple grid layout for Phase 1 implementation
        # Initialize SimpleTextureAllocator for space management
    
    def _evict_chunks(self, scale_level, needed_shape):
        # Remove least recently used chunks to free GPU space
        # Update allocation tracking and PyGFX texture contents
        # Continue until sufficient space available for new chunk

class SimpleTextureAllocator:
    def __init__(self, texture_size, chunk_shape):
        # Calculate grid dimensions for chunk placement
        # Initialize allocation bitmap for tracking used space
    
    def allocate(self, chunk_id, shape):
        # Find first available grid cell for chunk
        # Mark cell as allocated and calculate texture offset
        # Return TextureRegion with offset and size information
    
    def deallocate(self, chunk_id):
        # Free grid cell used by chunk
        # Update allocation bitmap to mark space as available
    
    def can_allocate(self, shape):
        # Check if space available for chunk of given shape
        # Used by eviction system to determine when to stop evicting
```

### 4.6 RenderManager Integration

**Design**: Enhanced RenderManager methods to handle chunked data responses and coordinate GPU upload via TextureAtlas.

**Key Considerations**:
- Receives numpy arrays from ChunkManager via existing event system
- Manages GPU upload and texture coordinate mapping
- Updates visuals with new texture regions
- Maintains clean separation between data and rendering concerns

```python
class RenderManager:  # Enhanced existing class
    def __init__(self):
        # Initialize existing RenderManager components
        # Add TextureAtlas for chunked rendering support
        self._texture_atlas = TextureAtlas(max_gpu_memory=1024*1024*1024)  # 1GB default
        self._chunk_gpu_cache = {}  # chunk_id -> TextureRegion
    
    def _on_new_slice(self, slice_data):
        # Handle both regular DataResponse and new ChunkedDataResponse
        if isinstance(slice_data, ChunkedDataResponse):
            self._handle_chunked_data(slice_data)
        else:
            # Existing logic for non-chunked data
            self._handle_regular_data(slice_data)
    
    def _handle_chunked_data(self, chunked_response):
        # Receive numpy arrays from ChunkManager
        # For each chunk in the response:
        #   - Upload to GPU via TextureAtlas
        #   - Get texture coordinates from TextureAtlas
        #   - Update visual with new texture region information
        # Emit canvas redraw request when updates complete
    
    def _update_visual_chunk(self, visual_id, chunk_id, texture_region):
        # Update specific visual with new chunk texture coordinates
        # Visual uses texture coordinates for shader-based rendering
        # Handle multiple chunks per visual for progressive loading
```

### 4.7 LoadingProgressTracker

**Design**: Provides status callback system for visual feedback about chunk loading progress. Tracks loading state per visual and emits events that can be consumed by UI components or visual renderers.

**Key Considerations**:
- Must be thread-safe since callbacks occur from background loading threads
- Should provide meaningful progress metrics (percentage complete, estimated time)
- Allow multiple visuals to register for callbacks independently
- Emit events compatible with existing Cellier event system

```python
class LoadingProgressTracker:
    def __init__(self):
        # Initialize thread-safe tracking for multiple visuals
        # Set up callback registration and event emission
        # Integrate with Cellier's existing event system via signals
    
    def register_visual(self, visual_id, callback):
        # Register callback function for visual's loading progress updates
        # Thread-safe registration allowing multiple callbacks per visual
    
    def start_loading_session(self, visual_id, total_chunks, current_scale):
        # Begin tracking new loading session for visual
        # Initialize progress counters and emit initial status
        # Thread-safe to handle concurrent loading from multiple scales
    
    def report_chunk_loaded(self, visual_id, chunk_id, success):
        # Update progress counters for successful or failed chunk loads
        # Calculate completion percentage based on loaded vs total chunks
        # Emit progress update to registered callbacks and event system
        # Thread-safe since called from background loading threads
    
    def _emit_update(self, visual_id):
        # Call all registered callbacks for visual with current status
        # Emit signal compatible with Cellier event system
        # Handle callback exceptions gracefully to avoid breaking loading
```

### 4.8 Data Structures

**Data structures to support the separation between data and rendering layers:**

```python
@dataclass
class ChunkedDataResponse(DataResponse):
    """Response containing CPU chunk data without GPU-specific information."""
    chunk_data: List[np.ndarray]  # CPU numpy arrays for each loaded chunk
    chunk_metadata: List[ChunkMetadata]  # IDs, priorities, scale levels
    loading_status: LoadingStatus
    missing_chunks: List[ChunkId]  # Still loading in background
    # NO texture coordinates - those are handled by RenderManager

@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: ChunkId
    scale_level: int
    priority: float
    world_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]

@dataclass
class TextureRegion:
    """GPU texture allocation information (rendering layer only)."""
    texture_id: str
    offset: Tuple[int, int, int]  # (x, y, z) offset within PyGFX texture
    size: Tuple[int, int, int]    # (width, height, depth) of region
    texture_coordinates: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
```