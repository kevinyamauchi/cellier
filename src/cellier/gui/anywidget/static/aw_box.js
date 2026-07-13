// Manager-rendered container: a tiny anywidget that asks the Jupyter widget
// manager to mount each child widget's view.  This keeps stock ipywidgets
// *controls* out while still composing multiple widgets in Jupyter, where any
// nesting of one widget inside another is a widget-manager operation.
//
// The `children` trait is a `widget_serialization` list, so the front-end
// receives reference strings ("IPY_MODEL_<id>"), not live models.  anywidget's
// generic model does not auto-deserialize them (the stock VBox/HBox register a
// deserializer; we cannot), so we resolve each reference via
// `widget_manager.get_model(id)` before creating its view.  Passing the raw
// string straight to `create_view` crashes inside the manager ("Cannot read
// properties of undefined (reading 'then')").
//
// We must NOT put the flex layout on `el`: `el` is the ipywidgets
// `DOMWidgetView` node, and ipywidgets applies the widget's `layout` trait to
// it -- with no `layout.display` set it writes `node.style.display = ''`, which
// wipes any `display:flex` we set during render (observed: nested boxes fell
// back to block, collapsing the grid to a vertical stack).  Instead we render
// into a private inner `<div>` we own; ipywidgets never touches it, so the flex
// layout survives at every nesting level.

async function render({ model, el }) {
  const box = document.createElement("div");
  box.className = "cellier-awbox";
  box.style.display = "flex";
  box.style.gap = "4px";
  box.style.flexDirection = model.get("direction") === "h" ? "row" : "column";
  // Cross-axis alignment (e.g. "center" to centre a fixed-width canvas over a
  // wider control panel); empty leaves the flexbox default (stretch).
  const align = model.get("align");
  if (align) box.style.alignItems = align;
  // Grow to fill available width but never below min_width (e.g. the
  // canvas+dims column, floored at the user's requested canvas_size);
  // 0 leaves the flexbox default (content-sized, no grow).
  const minWidth = model.get("min_width");
  if (minWidth) {
    box.style.flex = `1 1 ${minWidth}px`;
    box.style.minWidth = `${minWidth}px`;
  }
  el.appendChild(box);
  try {
    for (const ref of model.get("children") || []) {
      const id = String(ref).replace(/^IPY_MODEL_/, "");
      const childModel = await model.widget_manager.get_model(id);
      const view = await model.widget_manager.create_view(childModel);
      box.appendChild(view.el);
      // Give nested widgets (e.g. the rendercanvas canvas) the lifecycle event
      // they rely on to size themselves once attached to the DOM.
      view.trigger("displayed");
    }
  } catch (e) {
    // Surface failures in-cell instead of leaving a silently blank container.
    console.error("[AwBox] render failed", e);
    box.textContent = "AwBox error: " + (e && e.message);
  }
}

export default { render };
