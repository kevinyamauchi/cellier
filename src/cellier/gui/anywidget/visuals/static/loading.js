// Loading indicator ESM: a bar of target chunks resident over needed, and a
// status line.  Read-only; the Python side pushes every change.

function render({ model, el }) {
  el.classList.add("cellier-loading");

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";

  const body = document.createElement("div");
  body.className = "cellier-loading-body";

  const bar = document.createElement("progress");
  bar.className = "cellier-loading-bar";

  const text = document.createElement("span");
  text.className = "cellier-loading-text";

  function update() {
    label.textContent = model.get("title") || "Loading";
    bar.max = Math.max(model.get("maximum") ?? 1, 1);
    bar.value = model.get("value") ?? 0;
    text.textContent = model.get("text") ?? "";
    el.classList.toggle("cellier-loading-busy", !!model.get("busy"));
  }

  for (const name of ["title", "maximum", "value", "text", "busy"]) {
    model.on(`change:${name}`, update);
  }
  update();

  body.appendChild(bar);
  body.appendChild(text);
  row.appendChild(label);
  row.appendChild(body);
  el.appendChild(row);
}

export default { render };
