// Colormap control ESM (split out of the former composite panel.js).

function render({ model, el }) {
  el.classList.add("cellier-colormap-control");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = "Colormap";

  let cmNames = (model.get("colormap_names") || []).slice();
  const initCm = model.get("color_map") || "grays";
  if (!cmNames.includes(initCm)) cmNames = [initCm, ...cmNames];

  const select = document.createElement("select");
  select.className = "cellier-app-select";
  for (const name of cmNames) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  }
  select.value = initCm;

  select.addEventListener("change", () => {
    if (guard) return;
    model.set("color_map", select.value);
    model.save_changes();
  });

  model.on("change:color_map", () => {
    guard = true;
    try {
      const v = model.get("color_map") || "grays";
      if (!Array.from(select.options).some((o) => o.value === v)) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.prepend(opt);
      }
      select.value = v;
    } finally {
      guard = false;
    }
  });

  row.appendChild(label);
  row.appendChild(select);
  el.appendChild(row);
}

export default { render };
