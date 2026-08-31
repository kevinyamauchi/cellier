// Float number-input control ESM -- one asset shared by every unbounded float
// appearance field.
//
// Field-agnostic: generic `value` / `label` / `min` / `max` / `step` traits
// only. Unlike bounded_slider.js the bounds here are widget bounds someone
// picked rather than model constraints, so the input is a number box the user
// can type an exact value into rather than a slider.

function render({ model, el }) {
  el.classList.add("cellier-float-spin");

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

  applyBounds();
  applyValue();

  input.addEventListener("change", () => {
    if (guard) return;
    // A number input reports "" -- not the text -- when what the user typed
    // cannot be parsed, and Number("") is 0, so a Number.isNaN guard alone
    // lets a typo through as a real edit writing 0. Reject "" first.
    const raw = input.value;
    if (raw === "" || Number.isNaN(Number(raw))) {
      applyValue();
      return;
    }
    // Clamp to the declared bounds, so this control cannot emit a value its
    // Qt counterpart's spin box would have refused to produce.
    const value = Math.min(
      Math.max(Number(raw), Number(model.get("min"))),
      Number(model.get("max")),
    );
    if (value !== Number(raw)) applyLiteral(value);
    model.set("value", value);
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
  el.appendChild(row);
}

export default { render };
