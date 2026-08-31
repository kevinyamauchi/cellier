// Integer number-input control ESM -- one asset shared by every integer
// appearance field.
//
// Field-agnostic: generic `value` / `label` / `min` / `max` / `step` traits,
// plus `shuffle`. The shuffle button is a trait rather than a separate control
// type so this one asset still serves every integer field: it renders only
// where the number itself is meaningless and only the resulting difference
// matters (the labels colormap `salt`).

function render({ model, el }) {
  el.classList.add("cellier-int-spin");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = model.get("label") || "";

  const input = document.createElement("input");
  input.type = "number";
  input.className = "cellier-spin-input";

  function applyBounds() {
    input.min = String(model.get("min"));
    input.max = String(model.get("max"));
    input.step = String(model.get("step"));
  }

  function applyValue() {
    applyLiteral(model.get("value"));
  }

  function applyLiteral(value) {
    input.value = String(value);
  }

  function commit(value) {
    model.set("value", value);
    model.save_changes();
  }

  applyBounds();
  applyValue();

  input.addEventListener("change", () => {
    if (guard) return;
    // See float_spin.js: a number input reports "" for unparsable text, and
    // Number("") is 0, so the empty string has to be rejected explicitly or a
    // typo writes 0 as though the user had meant it.
    const raw = input.value;
    if (raw === "" || Number.isNaN(Number(raw))) {
      applyValue();
      return;
    }
    const value = Math.min(
      Math.max(Math.round(Number(raw)), Number(model.get("min"))),
      Number(model.get("max")),
    );
    if (value !== Math.round(Number(raw))) applyLiteral(value);
    commit(value);
  });

  row.appendChild(label);
  row.appendChild(input);

  if (model.get("shuffle")) {
    const button = document.createElement("button");
    button.className = "cellier-shuffle-button";
    button.textContent = "Shuffle";
    button.addEventListener("click", () => {
      const lo = Number(model.get("min"));
      const hi = Number(model.get("max"));
      commit(Math.floor(lo + Math.random() * (hi - lo + 1)));
    });
    row.appendChild(button);
  }

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

  el.appendChild(row);
}

export default { render };
