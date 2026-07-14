// AABB (bounding box) control ESM (split out of the former composite panel.js).

function render({ model, el }) {
  el.classList.add("cellier-aabb-widget");

  let guard = false;

  // ── enabled checkbox ─────────────────────────────────────────────────────
  const enabledRow = document.createElement("div");
  enabledRow.className = "cellier-app-row";
  const enabledCheck = document.createElement("input");
  enabledCheck.type = "checkbox";
  enabledCheck.checked = model.get("enabled") || false;
  const enabledLabel = document.createElement("label");
  enabledLabel.className = "cellier-app-label cellier-aabb-check-label";
  enabledLabel.textContent = "Show";
  enabledRow.appendChild(enabledLabel);
  enabledRow.appendChild(enabledCheck);
  el.appendChild(enabledRow);

  enabledCheck.addEventListener("change", () => {
    if (guard) return;
    model.set("enabled", enabledCheck.checked);
    model.save_changes();
  });
  model.on("change:enabled", () => {
    guard = true;
    try { enabledCheck.checked = model.get("enabled") || false; }
    finally { guard = false; }
  });

  // ── line width ───────────────────────────────────────────────────────────
  const lwRow = document.createElement("div");
  lwRow.className = "cellier-app-row";
  const lwLabel = document.createElement("label");
  lwLabel.className = "cellier-app-label";
  lwLabel.textContent = "Width";
  const lwInput = document.createElement("input");
  lwInput.type = "number";
  lwInput.className = "cellier-aabb-number";
  lwInput.min = 0.5;
  lwInput.max = 20;
  lwInput.step = 0.5;
  lwInput.value = model.get("line_width") || 2.0;
  lwRow.appendChild(lwLabel);
  lwRow.appendChild(lwInput);
  el.appendChild(lwRow);

  lwInput.addEventListener("change", () => {
    if (guard) return;
    model.set("line_width", parseFloat(lwInput.value));
    model.save_changes();
  });
  model.on("change:line_width", () => {
    guard = true;
    try { lwInput.value = model.get("line_width") || 2.0; }
    finally { guard = false; }
  });

  // ── color ────────────────────────────────────────────────────────────────
  const colorRow = document.createElement("div");
  colorRow.className = "cellier-app-row";
  const colorLabel = document.createElement("label");
  colorLabel.className = "cellier-app-label";
  colorLabel.textContent = "Color";
  const colorInput = document.createElement("input");
  colorInput.type = "color";
  colorInput.className = "cellier-aabb-color";
  colorInput.value = model.get("color") || "#ffffff";
  colorRow.appendChild(colorLabel);
  colorRow.appendChild(colorInput);
  el.appendChild(colorRow);

  colorInput.addEventListener("change", () => {
    if (guard) return;
    model.set("color", colorInput.value);
    model.save_changes();
  });
  model.on("change:color", () => {
    guard = true;
    try { colorInput.value = model.get("color") || "#ffffff"; }
    finally { guard = false; }
  });
}

export default { render };
