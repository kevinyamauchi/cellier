// Live slot: a column whose children can be replaced after it is rendered.
//
// Unlike `container.js` this mounts children through anywidget's composition
// API (`host.getWidget`), not the Jupyter widget manager's `create_view`.
// marimo implements `host` but not `create_view`, so this is what lets a dock
// change its contents on both hosts.  Children must therefore be anywidgets,
// serialized as `"anywidget:<model_id>"` references.
//
// Every mount gets its own AbortController.  Aborting it disposes the child
// views anywidget created for that mount, so replacing `children` tears the
// old views down before the new ones render -- and a mount superseded while it
// is still awaiting a child stops instead of appending stale views.

async function render({ model, el, host, signal }) {
  const heading = document.createElement("div");
  heading.className = "cellier-anywidget-box-title";

  const box = document.createElement("div");
  box.className = "cellier-anywidget-box cellier-anywidget-slot";
  box.style.display = "flex";
  box.style.flexDirection = "column";

  el.appendChild(heading);
  el.appendChild(box);

  let current = null;

  function applyChrome() {
    box.style.gap = `${model.get("gap")}px`;
    const title = model.get("title");
    heading.textContent = title || "";
    heading.hidden = !title;
  }

  async function mount() {
    if (current) current.abort();
    const controller = new AbortController();
    current = controller;
    const childSignal = signal
      ? AbortSignal.any([signal, controller.signal])
      : controller.signal;

    box.replaceChildren();
    try {
      for (const ref of model.get("children") || []) {
        const widget = await host.getWidget(ref);
        if (controller.signal.aborted) return;
        const slotEl = document.createElement("div");
        box.appendChild(slotEl);
        await widget.render({ el: slotEl, signal: childSignal });
        if (controller.signal.aborted) return;
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      console.error("[AnywidgetSlot] mount failed", e);
      box.textContent = "AnywidgetSlot error: " + (e && e.message);
    }
  }

  applyChrome();
  model.on("change:gap", applyChrome);
  model.on("change:title", applyChrome);
  model.on("change:children", mount);
  await mount();

  return () => {
    if (current) current.abort();
  };
}

export default { render };
