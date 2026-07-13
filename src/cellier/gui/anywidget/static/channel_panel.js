// Per-channel appearance panel ESM (self-contained; anywidget serves each
// _esm standalone, so there are no cross-file imports).
//
// The Python side adds one flattened synced trait per (channel, field):
// `ch{i}_{field}`.  Each control below binds to its trait exactly like the
// single-visual panel binds its appearance traits.  The local factory
// functions are adapted from panel.js's inline builders; each returns
// `{ el, set }` where `set` applies a value WITHOUT invoking its callback, so
// programmatic (model -> widget) writes are driven through `set` inside the
// `guard` flag.

const THROTTLE_MS = 50;

function makeCheckbox(initial) {
  const el = document.createElement("input");
  el.type = "checkbox";
  el.checked = Boolean(initial);
  return {
    el,
    set(value) {
      el.checked = Boolean(value);
    },
  };
}

function makeColormapSelect(names, initial) {
  let cmNames = (names || []).slice();
  if (initial && !cmNames.includes(initial)) cmNames = [initial, ...cmNames];

  const el = document.createElement("select");
  el.className = "cellier-app-select";
  for (const name of cmNames) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    el.appendChild(opt);
  }
  el.value = initial;

  return {
    el,
    set(value) {
      const v = value || "";
      if (!Array.from(el.options).some((o) => o.value === v)) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        el.prepend(opt);
      }
      el.value = v;
    },
  };
}

function makeFloatSlider(min, max, initial, { settledOnly, onChange }) {
  const el = document.createElement("div");
  el.className = "cellier-app-row-inner";

  const inp = document.createElement("input");
  inp.type = "range";
  inp.min = min;
  inp.max = max;
  inp.step = "any";
  inp.value = initial;

  const readout = document.createElement("span");
  readout.className = "cellier-app-readout";
  readout.textContent = Number(initial).toFixed(3);

  function emit() {
    readout.textContent = Number(inp.value).toFixed(3);
    onChange(parseFloat(inp.value));
  }

  if (!settledOnly) {
    inp.addEventListener("input", emit);
  }
  inp.addEventListener("change", emit);

  el.appendChild(inp);
  el.appendChild(readout);

  return {
    el,
    set(value) {
      inp.value = value;
      readout.textContent = Number(value).toFixed(3);
    },
  };
}

function makeClimSlider(range, initial, { onChange }) {
  const el = document.createElement("div");
  el.className = "cellier-clim-track";

  const rail = document.createElement("div");
  rail.className = "cellier-clim-rail";
  const fill = document.createElement("div");
  fill.className = "cellier-clim-fill";

  const loInput = document.createElement("input");
  loInput.type = "range";
  loInput.className = "cellier-clim-lo";
  loInput.min = range[0];
  loInput.max = range[1];
  loInput.step = "any";
  loInput.value = initial[0];

  const hiInput = document.createElement("input");
  hiInput.type = "range";
  hiInput.className = "cellier-clim-hi";
  hiInput.min = range[0];
  hiInput.max = range[1];
  hiInput.step = "any";
  hiInput.value = initial[1];

  function updateFill() {
    const mn = parseFloat(loInput.min);
    const mx = parseFloat(loInput.max);
    const span = mx - mn || 1;
    fill.style.left = (((parseFloat(loInput.value) - mn) / span) * 100) + "%";
    fill.style.right = (((mx - parseFloat(hiInput.value)) / span) * 100) + "%";
  }

  function submit() {
    onChange([parseFloat(loInput.value), parseFloat(hiInput.value)]);
  }

  // Throttle updates during a drag; flush on release.
  let timer = null;
  let pending = null;
  function schedule() {
    if (timer === null) {
      submit();
      pending = null;
      timer = setTimeout(function tick() {
        if (pending) {
          submit();
          pending = null;
          timer = setTimeout(tick, THROTTLE_MS);
        } else {
          timer = null;
        }
      }, THROTTLE_MS);
    } else {
      pending = true;
    }
  }

  loInput.addEventListener("input", () => {
    if (parseFloat(loInput.value) > parseFloat(hiInput.value)) {
      loInput.value = hiInput.value;
    }
    updateFill();
    schedule();
  });
  hiInput.addEventListener("input", () => {
    if (parseFloat(hiInput.value) < parseFloat(loInput.value)) {
      hiInput.value = loInput.value;
    }
    updateFill();
    schedule();
  });
  loInput.addEventListener("change", submit);
  hiInput.addEventListener("change", submit);

  el.appendChild(rail);
  el.appendChild(fill);
  el.appendChild(loInput);
  el.appendChild(hiInput);
  updateFill();

  return {
    el,
    set(value) {
      loInput.value = value[0];
      hiInput.value = value[1];
      updateFill();
    },
  };
}

function render({ model, el }) {
  el.classList.add("cellier-channel-panel");

  let guard = false;
  const fields = model.get("fields") || [];
  const labels = model.get("channel_labels") || {};
  const climRange = model.get("clim_range") || [0, 1];
  const cmNames = model.get("colormap_names") || [];

  // Channel indices are the keys of channel_labels (robust to sparse sets),
  // sorted numerically.
  const indices = Object.keys(labels)
    .map((k) => parseInt(k, 10))
    .sort((a, b) => a - b);

  for (const i of indices) {
    const group = document.createElement("div");
    group.className = "cellier-channel-row";

    const title = document.createElement("div");
    title.className = "cellier-section-title";
    title.textContent = labels[String(i)] || `Channel ${i}`;
    group.appendChild(title);

    for (const field of fields) {
      const trait = `ch${i}_${field}`;
      const row = document.createElement("div");
      row.className = "cellier-app-row";

      const label = document.createElement("label");
      label.className = "cellier-app-label";

      let control = null;

      if (field === "visible") {
        label.textContent = "Visible";
        control = makeCheckbox(model.get(trait));
        control.el.addEventListener("change", () => {
          if (guard) return;
          model.set(trait, control.el.checked);
          model.save_changes();
        });
      } else if (field === "color_map") {
        label.textContent = "Colormap";
        control = makeColormapSelect(cmNames, model.get(trait));
        control.el.addEventListener("change", () => {
          if (guard) return;
          model.set(trait, control.el.value);
          model.save_changes();
        });
      } else if (field === "clim") {
        label.textContent = "Contrast";
        control = makeClimSlider(climRange, model.get(trait), {
          onChange: (value) => {
            if (guard) return;
            model.set(trait, value);
            model.save_changes();
          },
        });
      } else if (field === "opacity") {
        label.textContent = "Opacity";
        control = makeFloatSlider(0.0, 1.0, model.get(trait), {
          settledOnly: false,
          onChange: (value) => {
            if (guard) return;
            model.set(trait, value);
            model.save_changes();
          },
        });
      } else if (field === "iso_threshold") {
        label.textContent = "Threshold";
        control = makeFloatSlider(0.0, 1.0, model.get(trait), {
          settledOnly: false,
          onChange: (value) => {
            if (guard) return;
            model.set(trait, value);
            model.save_changes();
          },
        });
      } else {
        continue;
      }

      // model -> widget: apply without re-emitting.
      model.on(`change:${trait}`, () => {
        guard = true;
        try {
          control.set(model.get(trait));
        } finally {
          guard = false;
        }
      });

      row.appendChild(label);
      row.appendChild(control.el);
      group.appendChild(row);
    }

    el.appendChild(group);
  }
}

export default { render };
