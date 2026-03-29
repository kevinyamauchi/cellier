# Events in cellier

## Overview
Cellier uses events to make sure that the model and view are kept in sync. Cellier has two main types of views. The controls, which are the GUI elements used to modify the state of the viewer model (e.g., selecting a colormap, or the dims sliders) and the renderer, which is a view of the data specified by the viewer model. All events are routed through the `EventBus` class.

## EventBus

The event bus serves as the interface between the model and the views. The models and the views connect their events to the `EventBus`. There are no direct connections between the models and the views. This allows for a clean separation of concerns and makes it easy to add new views or models without modifying existing code.

## Types of events
- visual model updates: these are emitted by the visual model when its state changes. 
- visual control updates: these are emitted by GUI elements that modify the state of the visual model.