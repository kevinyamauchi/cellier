// Colour control ESM -- one asset shared by every RGBA appearance field.
//
// Field-agnostic: the generic `value` trait, which is the model's full float
// RGBA, and `label`. <input type=color> is RGB-only in every browser, so alpha
// gets its own slider and the hex conversion happens here -- the Python side
// never speaks hex.
//
// Both sub-controls write the whole 4-list, so a change to either preserves
// the other. That also means this control emits one event per edit, like every
// other control type, rather than one per component.

function toHex(rgba) {
  const channel = (c) => {
    const clamped = Math.max(0, Math.min(1, Number(c) || 0));
    return Math.round(clamped * 255).toString(16).padStart(2, "0");
  };
  return `#${channel(rgba[0])}${channel(rgba[1])}${channel(rgba[2])}`;
}

function fromHex(hex) {
  const text = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(text.slice(i, i + 2), 16) / 255);
}

function render({ model, el }) {
  el.classList.add("cellier-color-picker");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = model.get("label") || "";

  const swatch = document.createElement("input");
  swatch.type = "color";
  swatch.className = "cellier-color-input";

  const alpha = document.createElement("input");
  alpha.type = "range";
  alpha.className = "cellier-alpha-input";
  alpha.min = "0";
  alpha.max = "1";
  alpha.step = "0.01";

  const readout = document.createElement("span");
  readout.className = "cellier-alpha-readout";

  function applyValue() {
    const rgba = model.get("value") || [1, 1, 1, 1];
    swatch.value = toHex(rgba);
    alpha.value = String(rgba[3]);
    readout.textContent = Number(rgba[3]).toFixed(2);
  }

  function commit(rgba) {
    model.set("value", rgba);
    model.save_changes();
  }

  applyValue();

  swatch.addEventListener("change", () => {
    if (guard) return;
    const current = model.get("value") || [1, 1, 1, 1];
    commit([...fromHex(swatch.value), Number(current[3])]);
  });

  alpha.addEventListener("input", () => {
    if (guard) return;
    readout.textContent = Number(alpha.value).toFixed(2);
  });

  alpha.addEventListener("change", () => {
    if (guard) return;
    const current = model.get("value") || [1, 1, 1, 1];
    commit([current[0], current[1], current[2], Number(alpha.value)]);
  });

  model.on("change:value", () => {
    guard = true;
    try {
      applyValue();
    } finally {
      guard = false;
    }
  });
  model.on("change:label", () => {
    label.textContent = model.get("label") || "";
  });

  row.appendChild(label);
  row.appendChild(swatch);
  row.appendChild(alpha);
  row.appendChild(readout);
  el.appendChild(row);
}

export default { render };
