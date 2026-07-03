// 2D/3D toggle button.  Each click bumps the synced `_clicks` counter, which
// the Python side observes to flip the viewer's displayed dimensions; the
// button relabels via the synced `label` trait.

function render({ model, el }) {
  const button = document.createElement("button");
  button.className = "cellier-dim-toggle";
  button.textContent = model.get("label");
  button.addEventListener("click", () => {
    model.set("_clicks", model.get("_clicks") + 1);
    model.save_changes();
  });
  model.on("change:label", () => {
    button.textContent = model.get("label");
  });
  el.appendChild(button);
}

export default { render };
