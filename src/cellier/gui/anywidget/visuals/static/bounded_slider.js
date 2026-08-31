// Bounded-slider control ESM -- one asset shared by every model-bounded float
// appearance field.
//
// Field-agnostic by construction: it reads the generic `value`, `label`,
// `min`, `max` and `step` traits, never a field-specific trait name, so every
// layer-3 class of this control type points `_esm` here (inherited from
// `AnywidgetBoundedSlider`).
//
// The range comes from the model's own ge/le, so a value this slider can
// produce is always one the model accepts.

function render({ model, el }) {
  el.classList.add("cellier-bounded-slider");

  // Suppresses the outbound listener while an inbound trait write is applied.
  // See toggle.js: on a plain <input> this is defensive, but the pattern stays
  // uniform across control types because it is load-bearing for any control
  // that re-dispatches on programmatic set.
  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = model.get("label") || "";

  const input = document.createElement("input");
  input.type = "range";
  input.className = "cellier-slider-input";

  const readout = document.createElement("span");
  readout.className = "cellier-slider-readout";

  function applyBounds() {
    input.min = String(model.get("min"));
    input.max = String(model.get("max"));
    input.step = String(model.get("step"));
  }

  function applyValue() {
    const value = model.get("value");
    input.value = String(value);
    readout.textContent = Number(value).toFixed(2);
  }

  applyBounds();
  applyValue();

  input.addEventListener("input", () => {
    if (guard) return;
    readout.textContent = Number(input.value).toFixed(2);
  });

  input.addEventListener("change", () => {
    if (guard) return;
    model.set("value", Number(input.value));
    model.save_changes();
  });

  model.on("change:value", () => {
    guard = true;
    try {
      applyValue();
    } finally {
      guard = false;
    }
  });
  model.on("change:min", applyBounds);
  model.on("change:max", applyBounds);
  model.on("change:step", applyBounds);
  model.on("change:label", () => {
    label.textContent = model.get("label") || "";
  });

  row.appendChild(label);
  row.appendChild(input);
  row.appendChild(readout);
  el.appendChild(row);
}

export default { render };
