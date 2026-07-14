// Volume render controls ESM: mode select + mode-dependent ISO threshold /
// attenuation rows (split out of the former composite panel.js).

function render({ model, el }) {
  el.classList.add("cellier-volume-render-controls");

  let guard = false;
  let isoRow = null;
  let attenuationRow = null;

  const initMode = model.get("render_mode") || "mip";

  const modeRow = document.createElement("div");
  modeRow.className = "cellier-app-row";

  const modeLabel = document.createElement("label");
  modeLabel.className = "cellier-app-label";
  modeLabel.textContent = "Mode";

  const renderModeSelect = document.createElement("select");
  renderModeSelect.className = "cellier-app-select";
  for (const opt of ["iso", "smooth_iso", "mip", "attenuated_mip"]) {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    renderModeSelect.appendChild(o);
  }
  renderModeSelect.value = initMode;

  renderModeSelect.addEventListener("change", () => {
    if (guard) return;
    model.set("render_mode", renderModeSelect.value);
    model.save_changes();
    applyModeVisibility(renderModeSelect.value);
  });

  model.on("change:render_mode", () => {
    guard = true;
    try {
      const v = model.get("render_mode") || "mip";
      renderModeSelect.value = v;
      applyModeVisibility(v);
    } finally {
      guard = false;
    }
  });

  modeRow.appendChild(modeLabel);
  modeRow.appendChild(renderModeSelect);
  el.appendChild(modeRow);

  // Helper: build a labeled float range row and wire it.
  function buildFloatRow(labelText, traitName, minVal, maxVal, settledOnly) {
    const row = document.createElement("div");
    row.className = "cellier-app-row";

    const lbl = document.createElement("label");
    lbl.className = "cellier-app-label";
    lbl.textContent = labelText;

    const initVal = model.get(traitName);
    const inp = document.createElement("input");
    inp.type = "range";
    inp.min = minVal;
    inp.max = maxVal;
    inp.step = "any";
    inp.value = initVal;

    const readout = document.createElement("span");
    readout.className = "cellier-app-readout";
    readout.textContent = Number(initVal).toFixed(3);

    function emit() {
      if (guard) return;
      readout.textContent = Number(inp.value).toFixed(3);
      model.set(traitName, parseFloat(inp.value));
      model.save_changes();
    }

    if (!settledOnly) {
      inp.addEventListener("input", emit);
    }
    inp.addEventListener("change", emit);

    model.on("change:" + traitName, () => {
      guard = true;
      try {
        const v = model.get(traitName);
        inp.value = v;
        readout.textContent = Number(v).toFixed(3);
      } finally {
        guard = false;
      }
    });

    row.appendChild(lbl);
    row.appendChild(inp);
    row.appendChild(readout);
    el.appendChild(row);
    return row;
  }

  isoRow = buildFloatRow("Threshold", "iso_threshold", 0.0, 1.0, false);
  attenuationRow = buildFloatRow("Attenuation", "attenuation", 0.0, 10.0, false);

  function applyModeVisibility(mode) {
    isoRow.style.display = mode === "iso" || mode === "smooth_iso" ? "" : "none";
    attenuationRow.style.display = mode === "attenuated_mip" ? "" : "none";
  }

  applyModeVisibility(initMode);
}

export default { render };
