// Render-settings anywidget ESM.  One file drives all three panels
// (outlines, ambient occlusion, temporal accumulation): the Python side
// syncs a `controls` spec and one scalar trait per control, so the shape of
// a panel is data rather than three near-identical renderers.
//
// Sliders throttle while dragging; every other input writes on change.  The
// throttle matters because a few of these fields recompile a shader.

const THROTTLE_MS = 50;

function render({ model, el }) {
  el.classList.add("cellier-render-panel");

  // Set while pushing a model value into an input, so the input's own
  // change handler does not write it straight back.
  let guard = false;

  const heading = document.createElement("div");
  heading.className = "cellier-render-title";
  heading.textContent = model.get("title");
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-render-body";
  el.appendChild(body);

  const readoutBox = document.createElement("div");
  readoutBox.className = "cellier-render-readouts";
  el.appendChild(readoutBox);

  const actionButton = document.createElement("button");
  actionButton.className = "cellier-render-action";
  actionButton.addEventListener("click", () => {
    model.set("_action_clicks", model.get("_action_clicks") + 1);
    model.save_changes();
  });
  el.appendChild(actionButton);

  // trait name -> function pushing a model value into the input.
  const appliers = {};
  // Leading + trailing throttle, shared across sliders: a user drags one at
  // a time.
  let timer = null;
  let pending = null;

  function write(trait, value) {
    if (guard) return;
    model.set(trait, value);
    model.save_changes();
  }

  function scheduleWrite(trait, value) {
    if (guard) return;
    if (timer === null) {
      write(trait, value); // leading edge
      pending = null;
      timer = setTimeout(function tick() {
        if (pending !== null) {
          write(pending.trait, pending.value);
          pending = null;
          timer = setTimeout(tick, THROTTLE_MS);
        } else {
          timer = null;
        }
      }, THROTTLE_MS);
    } else {
      pending = { trait, value };
    }
  }

  function makeRow(spec, control) {
    const row = document.createElement("div");
    row.className = "cellier-render-row";
    // An empty label means the group around the control already names it,
    // and a row repeating that would be a line saying nothing.
    if (spec.label) {
      const label = document.createElement("label");
      label.className = "cellier-render-label";
      label.textContent = spec.label;
      if (spec.tooltip) label.title = spec.tooltip;
      row.appendChild(label);
    }
    row.appendChild(control);
    return row;
  }

  function buildBool(spec) {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(model.get(spec.trait));
    if (spec.tooltip) input.title = spec.tooltip;
    input.addEventListener("change", () => write(spec.trait, input.checked));
    appliers[spec.trait] = (value) => {
      input.checked = Boolean(value);
    };

    // A checkbox names itself, rather than being named by a row.
    const row = document.createElement("div");
    row.className = "cellier-render-row cellier-render-check";
    const label = document.createElement("label");
    label.className = "cellier-render-checklabel";
    if (spec.tooltip) label.title = spec.tooltip;
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + spec.label));
    row.appendChild(label);
    return row;
  }

  function buildNumber(spec) {
    const wrap = document.createElement("div");
    wrap.className = "cellier-render-numeric";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step;

    const readout = document.createElement("span");
    readout.className = "cellier-render-value";

    // A null value means "derive this from the scene" (the occlusion
    // radius).  There is no separate flag, so the readout is where that
    // state is visible.
    function show(value) {
      if (value === null || value === undefined) {
        readout.textContent = "auto";
        slider.disabled = true;
        return;
      }
      slider.disabled = false;
      slider.value = value;
      readout.textContent = spec.kind === "int" ? String(value) : Number(value).toPrecision(3);
    }

    slider.addEventListener("input", () => {
      const value = spec.kind === "int" ? parseInt(slider.value, 10) : parseFloat(slider.value);
      readout.textContent = spec.kind === "int" ? String(value) : value.toPrecision(3);
      scheduleWrite(spec.trait, value);
    });
    if (spec.tooltip) {
      slider.title = spec.tooltip;
      readout.title = spec.tooltip;
    }

    show(model.get(spec.trait));
    appliers[spec.trait] = show;

    wrap.appendChild(slider);
    wrap.appendChild(readout);
    return makeRow(spec, wrap);
  }

  function buildColor(spec) {
    // Two rows, mirroring the Qt control: alpha needs a control of its own
    // because every colour here is float RGBA and a colour input is
    // RGB-only, and it needs a row of its own because sharing one left the
    // slider a few pixels wide.  The swatch opens the picker by itself --
    // an <input type=color> already is the button.
    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.className = "cellier-render-colorswatch";

    const alpha = document.createElement("input");
    alpha.type = "range";
    alpha.min = 0;
    alpha.max = 1;
    alpha.step = 0.01;
    alpha.className = "cellier-render-alpha";

    const alphaReadout = document.createElement("span");
    alphaReadout.className = "cellier-render-value";

    function toHex(rgba) {
      const byte = (c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0");
      return "#" + byte(rgba[0]) + byte(rgba[1]) + byte(rgba[2]);
    }
    function fromHex(hex, a) {
      const text = hex.replace("#", "");
      return [
        parseInt(text.slice(0, 2), 16) / 255,
        parseInt(text.slice(2, 4), 16) / 255,
        parseInt(text.slice(4, 6), 16) / 255,
        a,
      ];
    }

    function emit() {
      write(spec.trait, fromHex(swatch.value, parseFloat(alpha.value)));
    }
    swatch.addEventListener("change", emit);
    alpha.addEventListener("change", emit);
    alpha.addEventListener("input", () => {
      alphaReadout.textContent = Number(alpha.value).toFixed(2);
    });

    function show(rgba) {
      swatch.value = toHex(rgba);
      const a = rgba.length > 3 ? rgba[3] : 1.0;
      alpha.value = a;
      alphaReadout.textContent = Number(a).toFixed(2);
    }
    show(model.get(spec.trait) || [1, 1, 1, 1]);
    appliers[spec.trait] = show;

    if (spec.tooltip) swatch.title = spec.tooltip;

    const alphaWrap = document.createElement("div");
    alphaWrap.className = "cellier-render-numeric";
    alphaWrap.appendChild(alpha);
    alphaWrap.appendChild(alphaReadout);

    const rows = document.createDocumentFragment();
    rows.appendChild(makeRow(spec, swatch));
    rows.appendChild(makeRow({ label: "Alpha", tooltip: "" }, alphaWrap));
    return rows;
  }

  function buildPalette(spec) {
    // The selection palette: one swatch per slot, plus add and remove.
    //
    // The length is editable because a per-visual control offers these
    // entries as its choice set -- so the palette length is the number of
    // groups a user can tell apart, not a configuration detail.  Removal
    // takes the LAST entry only: removing from the middle renumbers every
    // slot above it and silently recolours visuals that never changed.
    //
    // Alpha is carried through untouched; a translucent selection outline
    // reads as a rendering error rather than as a choice.
    const wrap = document.createElement("div");
    wrap.className = "cellier-render-palette";

    const swatchRow = document.createElement("div");
    swatchRow.className = "cellier-render-swatches";
    wrap.appendChild(swatchRow);

    const buttons = document.createElement("div");
    buttons.className = "cellier-render-palette-buttons";
    const addButton = document.createElement("button");
    addButton.textContent = "+";
    addButton.title = "Add a slot (at most " + spec.max_slots + ")";
    const removeButton = document.createElement("button");
    removeButton.textContent = "-";
    removeButton.title = "Remove the last slot";
    buttons.appendChild(addButton);
    buttons.appendChild(removeButton);
    wrap.appendChild(buttons);

    let entries = (model.get(spec.trait) || []).map((e) => e.slice());

    function toHex(rgba) {
      const byte = (c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0");
      return "#" + byte(rgba[0]) + byte(rgba[1]) + byte(rgba[2]);
    }

    function draw() {
      swatchRow.textContent = "";
      const usage = model.get("slot_usage") || {};
      entries.forEach((rgba, index) => {
        const slot = index + 1;
        const cell = document.createElement("div");
        cell.className = "cellier-render-slot";

        const swatch = document.createElement("input");
        swatch.type = "color";
        swatch.className = "cellier-render-swatch";
        swatch.value = toHex(rgba);
        const count = usage[String(slot)] || 0;
        swatch.title = count
          ? "Slot " + slot + " -- used by " + count + " visual(s)"
          : "Slot " + slot + " -- unused";
        swatch.addEventListener("change", () => {
          const text = swatch.value.replace("#", "");
          const alpha = entries[index].length > 3 ? entries[index][3] : 1.0;
          entries[index] = [
            parseInt(text.slice(0, 2), 16) / 255,
            parseInt(text.slice(2, 4), 16) / 255,
            parseInt(text.slice(4, 6), 16) / 255,
            alpha,
          ];
          write(spec.trait, entries.map((e) => e.slice()));
        });
        cell.appendChild(swatch);

        // The slot number is what connects this row to `visual.outline.slot`
        // and to the warnings, which both speak in numbers.
        const caption = document.createElement("span");
        caption.className = "cellier-render-slot-number";
        caption.textContent = count ? slot + " x" + count : String(slot);
        cell.appendChild(caption);

        swatchRow.appendChild(cell);
      });
    }

    addButton.addEventListener("click", () => {
      if (entries.length >= spec.max_slots) return;
      entries = entries.concat([spec.next_color]);
      draw();
      write(spec.trait, entries.map((e) => e.slice()));
    });
    removeButton.addEventListener("click", () => {
      if (entries.length <= 1) return;
      entries = entries.slice(0, -1);
      draw();
      write(spec.trait, entries.map((e) => e.slice()));
    });

    appliers[spec.trait] = (value) => {
      entries = (value || []).map((e) => e.slice());
      draw();
    };

    draw();
    if (spec.tooltip) wrap.title = spec.tooltip;
    return makeRow(spec, wrap);
  }

  function build() {
    body.textContent = "";
    for (const key of Object.keys(appliers)) delete appliers[key];

    const specs = model.get("controls") || [];
    // Consecutive controls sharing a group heading are drawn inside one
    // titled box: rows grouped together are ones that only mean something
    // together.
    let currentGroup = null;
    let target = body;

    for (const spec of specs) {
      const group = spec.group || "";
      if (group !== currentGroup) {
        currentGroup = group;
        if (group === "") {
          target = body;
        } else {
          const box = document.createElement("fieldset");
          box.className = "cellier-render-group";
          const legend = document.createElement("legend");
          legend.textContent = group;
          box.appendChild(legend);
          body.appendChild(box);
          target = box;
        }
      }
      if (spec.kind === "bool") target.appendChild(buildBool(spec));
      else if (spec.kind === "color") target.appendChild(buildColor(spec));
      else if (spec.kind === "palette") target.appendChild(buildPalette(spec));
      else target.appendChild(buildNumber(spec));
    }

    for (const spec of specs) {
      model.on("change:" + spec.trait, () => {
        const applier = appliers[spec.trait];
        if (!applier) return;
        guard = true;
        try {
          applier(model.get(spec.trait));
        } finally {
          guard = false;
        }
      });
    }
  }

  function updateReadouts() {
    readoutBox.textContent = "";
    for (const [label, text] of model.get("readouts") || []) {
      const row = document.createElement("div");
      row.className = "cellier-render-readout";
      const name = document.createElement("span");
      name.className = "cellier-render-label";
      name.textContent = label;
      const value = document.createElement("span");
      value.textContent = text;
      row.appendChild(name);
      row.appendChild(value);
      readoutBox.appendChild(row);
    }
  }

  function updateAction() {
    const label = model.get("action_label");
    actionButton.textContent = label;
    actionButton.style.display = label ? "" : "none";
  }

  build();
  updateReadouts();
  updateAction();
  model.on("change:controls", build);
  model.on("change:slot_usage", build);
  model.on("change:readouts", updateReadouts);
  model.on("change:action_label", updateAction);
  model.on("change:title", () => {
    heading.textContent = model.get("title");
  });

  return () => {
    if (timer !== null) clearTimeout(timer);
  };
}

export default { render };
