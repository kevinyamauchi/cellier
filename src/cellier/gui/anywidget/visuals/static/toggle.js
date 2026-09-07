// Toggle control ESM -- one asset shared by every boolean appearance field.
//
// Deliberately field-agnostic: it reads the generic `value` and `label`
// traits, never a field-specific trait name, so `AnywidgetVisibleToggle`,
// `AnywidgetWireframeToggle` and every other layer-3 class point `_esm` at
// this one file (they inherit it from `AnywidgetToggle`).

function render({ model, el }) {
  el.classList.add("cellier-toggle");

  // Suppresses the outbound `change` listener while an inbound trait write is
  // being applied. A programmatic `input.checked = ...` fires no event, so on
  // a plain <input> this is defensive rather than load-bearing -- but any
  // control type that re-dispatches on programmatic set needs it, so the
  // pattern stays uniform across control types.
  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = model.get("label") || "";

  const input = document.createElement("input");
  input.type = "checkbox";
  input.className = "cellier-toggle-input";
  input.checked = model.get("value") || false;

  input.addEventListener("change", () => {
    if (guard) return;
    model.set("value", input.checked);
    model.save_changes();
  });

  model.on("change:value", () => {
    guard = true;
    try {
      input.checked = model.get("value") || false;
    } finally {
      guard = false;
    }
  });

  model.on("change:label", () => {
    label.textContent = model.get("label") || "";
  });

  row.appendChild(label);
  row.appendChild(input);
  el.appendChild(row);
}

export default { render };
