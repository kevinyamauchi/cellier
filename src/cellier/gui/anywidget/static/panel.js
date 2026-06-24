// Composite control panel ESM (Design A).
//
// Phase 1: one labeled range input per axis, driven by the synced traits
// `axis_ranges`, `axis_labels`, `slice_indices`, and the visibility lists
// `displayed_axes` / `stacked_axes` / `non_displayed`.
//
// Echo suppression: a JS `guard` flag wraps programmatic (model -> widget)
// input updates so they do not re-emit.
//
// Live updates: dragging emits throttled updates on `input` (not just on the
// settled `change` event) so the slice reslices continuously while dragging.
// This mirrors QtDimsSliders' 50 ms rate-limiter (a leading-edge submit plus a
// trailing tick), keeping one reslice per window rather than one per pixel --
// important here because each `save_changes` is a kernel round-trip.

const THROTTLE_MS = 50;

function render({ model, el }) {
  el.classList.add("cellier-control-panel");
  const container = document.createElement("div");
  container.className = "cellier-dims";
  el.appendChild(container);

  let guard = false;
  let rows = {}; // axis(str) -> { row, input, readout }

  // Leading + trailing throttle shared across axes (a user drags one at a
  // time).  `submit` writes immediately; `scheduleSubmit` rate-limits drags.
  let timer = null;
  let pending = null; // { axis, value } captured during the throttle window

  function submit(axis, value) {
    if (guard) return;
    const current = { ...(model.get("slice_indices") || {}) };
    current[axis] = Number(value);
    model.set("slice_indices", current);
    model.save_changes();
  }

  function scheduleSubmit(axis, value) {
    if (guard) return;
    if (timer === null) {
      submit(axis, value); // leading edge
      pending = null;
      timer = setTimeout(function tick() {
        if (pending !== null) {
          submit(pending.axis, pending.value);
          pending = null;
          timer = setTimeout(tick, THROTTLE_MS);
        } else {
          timer = null;
        }
      }, THROTTLE_MS);
    } else {
      pending = { axis, value }; // coalesce to the latest within the window
    }
  }

  function build() {
    container.innerHTML = "";
    rows = {};
    const ranges = model.get("axis_ranges") || {};
    const labels = model.get("axis_labels") || {};
    const slices = model.get("slice_indices") || {};

    for (const axis of Object.keys(ranges)) {
      const bounds = ranges[axis] || [0, 0];
      const lo = bounds[0];
      const hi = bounds[1];

      const row = document.createElement("div");
      row.className = "cellier-dim-row";

      const label = document.createElement("label");
      label.className = "cellier-dim-label";
      label.textContent = labels[axis] !== undefined ? labels[axis] : axis;

      const input = document.createElement("input");
      input.type = "range";
      input.min = lo;
      input.max = hi;
      input.step = 1;
      input.value = slices[axis] !== undefined ? slices[axis] : lo;

      const readout = document.createElement("span");
      readout.className = "cellier-dim-readout";
      readout.textContent = input.value;

      input.addEventListener("input", () => {
        readout.textContent = input.value;
        scheduleSubmit(axis, input.value); // live, throttled
      });
      input.addEventListener("change", () => {
        // Final flush on release so the last position always lands even if it
        // arrived between throttle ticks.
        pending = null;
        submit(axis, input.value);
      });

      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(readout);
      container.appendChild(row);
      rows[axis] = { row, input, readout };
    }
    updateVisibility();
    syncValues();
  }

  function updateVisibility() {
    const displayed = (model.get("displayed_axes") || []).map(String);
    const stacked = (model.get("stacked_axes") || []).map(String);
    const nondisp = (model.get("non_displayed") || []).map(String);
    for (const axis of Object.keys(rows)) {
      const hidden =
        displayed.includes(axis) ||
        stacked.includes(axis) ||
        nondisp.includes(axis);
      rows[axis].row.style.display = hidden ? "none" : "";
    }
  }

  function syncValues() {
    guard = true;
    try {
      const slices = model.get("slice_indices") || {};
      for (const axis of Object.keys(rows)) {
        if (Object.prototype.hasOwnProperty.call(slices, axis)) {
          rows[axis].input.value = slices[axis];
          rows[axis].readout.textContent = slices[axis];
        }
      }
    } finally {
      guard = false;
    }
  }

  build();
  model.on("change:axis_ranges", build);
  model.on("change:axis_labels", build);
  model.on("change:slice_indices", syncValues);
  model.on("change:displayed_axes", updateVisibility);
  model.on("change:stacked_axes", updateVisibility);
  model.on("change:non_displayed", updateVisibility);
}

export default { render };
