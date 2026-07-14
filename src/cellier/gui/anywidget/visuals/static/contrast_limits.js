// Contrast-limits (dual-range) control ESM (split out of the former
// composite panel.js).  Emits throttled updates on drag, flushes on release.

const THROTTLE_MS = 50;

function render({ model, el }) {
  el.classList.add("cellier-clim-slider");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = "Contrast";

  const climRange = model.get("clim_range") || [0, 1];
  const initClim = model.get("clim") || [0, 1];

  const track = document.createElement("div");
  track.className = "cellier-clim-track";

  const rail = document.createElement("div");
  rail.className = "cellier-clim-rail";

  const fill = document.createElement("div");
  fill.className = "cellier-clim-fill";

  const loInput = document.createElement("input");
  loInput.type = "range";
  loInput.className = "cellier-clim-lo";
  loInput.min = climRange[0];
  loInput.max = climRange[1];
  loInput.step = "any";
  loInput.value = initClim[0];

  const hiInput = document.createElement("input");
  hiInput.type = "range";
  hiInput.className = "cellier-clim-hi";
  hiInput.min = climRange[0];
  hiInput.max = climRange[1];
  hiInput.step = "any";
  hiInput.value = initClim[1];

  const readout = document.createElement("span");
  readout.className = "cellier-app-readout cellier-clim-readout";
  readout.textContent = fmtClim(initClim[0], initClim[1]);

  function fmtClim(lo, hi) {
    return `${fmtNum(lo)} – ${fmtNum(hi)}`;
  }
  function fmtNum(v) {
    return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(0);
  }

  function updateFill() {
    const mn = parseFloat(loInput.min);
    const mx = parseFloat(loInput.max);
    const span = mx - mn || 1;
    fill.style.left = ((parseFloat(loInput.value) - mn) / span * 100) + "%";
    fill.style.right = ((mx - parseFloat(hiInput.value)) / span * 100) + "%";
  }

  function submitClim() {
    if (guard) return;
    const lo = parseFloat(loInput.value);
    const hi = parseFloat(hiInput.value);
    readout.textContent = fmtClim(lo, hi);
    model.set("clim", [lo, hi]);
    model.save_changes();
  }

  let climTimer = null;
  let climPending = null;

  function scheduleClimSubmit() {
    if (guard) return;
    if (climTimer === null) {
      submitClim();
      climPending = null;
      climTimer = setTimeout(function tick() {
        if (climPending) {
          submitClim();
          climPending = null;
          climTimer = setTimeout(tick, THROTTLE_MS);
        } else {
          climTimer = null;
        }
      }, THROTTLE_MS);
    } else {
      climPending = true;
    }
  }

  loInput.addEventListener("input", () => {
    if (guard) return;
    if (parseFloat(loInput.value) > parseFloat(hiInput.value)) {
      loInput.value = hiInput.value;
    }
    readout.textContent = fmtClim(parseFloat(loInput.value), parseFloat(hiInput.value));
    updateFill();
    scheduleClimSubmit();
  });
  hiInput.addEventListener("input", () => {
    if (guard) return;
    if (parseFloat(hiInput.value) < parseFloat(loInput.value)) {
      hiInput.value = loInput.value;
    }
    readout.textContent = fmtClim(parseFloat(loInput.value), parseFloat(hiInput.value));
    updateFill();
    scheduleClimSubmit();
  });
  loInput.addEventListener("change", () => { if (!guard) submitClim(); });
  hiInput.addEventListener("change", () => { if (!guard) submitClim(); });

  model.on("change:clim", () => {
    guard = true;
    try {
      const v = model.get("clim") || [0, 1];
      loInput.value = v[0];
      hiInput.value = v[1];
      readout.textContent = fmtClim(v[0], v[1]);
      updateFill();
    } finally {
      guard = false;
    }
  });

  track.appendChild(rail);
  track.appendChild(fill);
  track.appendChild(loInput);
  track.appendChild(hiInput);
  updateFill();

  row.appendChild(label);
  row.appendChild(track);
  row.appendChild(readout);
  el.appendChild(row);
}

export default { render };
