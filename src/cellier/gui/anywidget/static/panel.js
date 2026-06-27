// Composite control panel ESM (Design A).
//
// Phase 1: one labeled range input per axis, driven by the synced traits
// `axis_ranges`, `axis_labels`, `slice_indices`, and the visibility lists
// `displayed_axes` / `stacked_axes` / `non_displayed`.
//
// Phase 2: appearance section (colormap, clim, render mode, iso threshold,
// attenuation, lod bias), AABB section (enabled, line width, color), and
// an optional dataset-info detail block.
//
// Echo suppression: a JS `guard` flag wraps programmatic (model -> widget)
// input updates so they do not re-emit.
//
// Live updates: dragging dims sliders and clim sliders emits throttled updates
// on `input` (not just the settled `change` event).  LOD-bias and render-mode
// emit only on `change` to avoid triggering unnecessary reslices.

const THROTTLE_MS = 50;

function render({ model, el }) {
  el.classList.add("cellier-control-panel");

  let guard = false;

  // ── Appearance section ────────────────────────────────────────────────────

  if (model.get("has_appearance")) {
    buildAppearance();
  }

  function buildAppearance() {
    const fields = new Set(model.get("appearance_fields") || []);

    const section = document.createElement("div");
    section.className = "cellier-appearance";

    const title = document.createElement("div");
    title.className = "cellier-section-title";
    title.textContent = "Appearance";
    section.appendChild(title);

    // -- Colormap --------------------------------------------------------
    if (fields.has("color_map")) {
      let cmNames = (model.get("colormap_names") || []).slice();
      const initCm = model.get("color_map") || "grays";
      if (!cmNames.includes(initCm)) cmNames = [initCm, ...cmNames];

      const cmRow = document.createElement("div");
      cmRow.className = "cellier-app-row";

      const cmLabel = document.createElement("label");
      cmLabel.className = "cellier-app-label";
      cmLabel.textContent = "Colormap";

      const cmSelect = document.createElement("select");
      cmSelect.className = "cellier-app-select";
      for (const name of cmNames) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        cmSelect.appendChild(opt);
      }
      cmSelect.value = initCm;

      cmSelect.addEventListener("change", () => {
        if (guard) return;
        model.set("color_map", cmSelect.value);
        model.save_changes();
      });

      model.on("change:color_map", () => {
        guard = true;
        try {
          const v = model.get("color_map") || "grays";
          if (!Array.from(cmSelect.options).some((o) => o.value === v)) {
            const opt = document.createElement("option");
            opt.value = v;
            opt.textContent = v;
            cmSelect.prepend(opt);
          }
          cmSelect.value = v;
        } finally {
          guard = false;
        }
      });

      cmRow.appendChild(cmLabel);
      cmRow.appendChild(cmSelect);
      section.appendChild(cmRow);
    }

    // -- Contrast limits (dual-range) ------------------------------------
    if (fields.has("clim")) {
      const climRange = model.get("clim_range") || [0, 1];
      const initClim = model.get("clim") || [0, 1];

      const climRow = document.createElement("div");
      climRow.className = "cellier-app-row";

      const climLabel = document.createElement("label");
      climLabel.className = "cellier-app-label";
      climLabel.textContent = "Contrast";

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

      const climReadout = document.createElement("span");
      climReadout.className = "cellier-app-readout cellier-clim-readout";
      climReadout.textContent = fmtClim(initClim[0], initClim[1]);

      function fmtClim(lo, hi) {
        return `${fmtNum(lo)} – ${fmtNum(hi)}`;
      }
      function fmtNum(v) {
        // Use fixed-3 for small ranges, otherwise integer
        return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(0);
      }

      function updateFill() {
        const mn = parseFloat(loInput.min);
        const mx = parseFloat(loInput.max);
        const span = mx - mn || 1;
        fill.style.left  = ((parseFloat(loInput.value) - mn) / span * 100) + "%";
        fill.style.right = ((mx - parseFloat(hiInput.value)) / span * 100) + "%";
      }

      function submitClim() {
        if (guard) return;
        const lo = parseFloat(loInput.value);
        const hi = parseFloat(hiInput.value);
        climReadout.textContent = fmtClim(lo, hi);
        model.set("clim", [lo, hi]);
        model.save_changes();
      }

      // Throttle clim updates like dims sliders
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
        // clamp lo <= hi
        if (parseFloat(loInput.value) > parseFloat(hiInput.value)) {
          loInput.value = hiInput.value;
        }
        climReadout.textContent = fmtClim(parseFloat(loInput.value), parseFloat(hiInput.value));
        updateFill();
        scheduleClimSubmit();
      });
      hiInput.addEventListener("input", () => {
        if (guard) return;
        // clamp hi >= lo
        if (parseFloat(hiInput.value) < parseFloat(loInput.value)) {
          hiInput.value = loInput.value;
        }
        climReadout.textContent = fmtClim(parseFloat(loInput.value), parseFloat(hiInput.value));
        updateFill();
        scheduleClimSubmit();
      });
      // Final flush on release
      loInput.addEventListener("change", () => { if (!guard) submitClim(); });
      hiInput.addEventListener("change", () => { if (!guard) submitClim(); });

      model.on("change:clim", () => {
        guard = true;
        try {
          const v = model.get("clim") || [0, 1];
          loInput.value = v[0];
          hiInput.value = v[1];
          climReadout.textContent = fmtClim(v[0], v[1]);
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
      climRow.appendChild(climLabel);
      climRow.appendChild(track);
      climRow.appendChild(climReadout);
      section.appendChild(climRow);
    }

    // -- Render mode (+ mode-dependent rows) -----------------------------
    let isoRow = null;
    let attenuationRow = null;
    let renderModeSelect = null;

    if (fields.has("render_mode")) {
      const initMode = model.get("render_mode") || "mip";

      const modeRow = document.createElement("div");
      modeRow.className = "cellier-app-row";

      const modeLabel = document.createElement("label");
      modeLabel.className = "cellier-app-label";
      modeLabel.textContent = "Mode";

      renderModeSelect = document.createElement("select");
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
      section.appendChild(modeRow);

      // -- ISO threshold (visible for iso / smooth_iso) ------------------
      if (fields.has("iso_threshold")) {
        isoRow = buildFloatRow(
          section,
          "Threshold",
          "iso_threshold",
          model.get("iso_threshold") || 0.2,
          0.0,
          1.0,
          false // emit live (no reslice for threshold)
        );
      }

      // -- Attenuation (visible for attenuated_mip) ----------------------
      if (fields.has("attenuation")) {
        attenuationRow = buildFloatRow(
          section,
          "Attenuation",
          "attenuation",
          model.get("attenuation") || 1.0,
          0.0,
          10.0,
          false
        );
      }

      applyModeVisibility(initMode);
    } else if (fields.has("iso_threshold")) {
      // iso_threshold without render_mode (always visible)
      buildFloatRow(
        section,
        "Threshold",
        "iso_threshold",
        model.get("iso_threshold") || 0.2,
        0.0,
        1.0,
        false
      );
    }

    // -- LOD bias (emit only on change to avoid excessive reslices) ------
    if (fields.has("lod_bias")) {
      buildFloatRow(
        section,
        "LOD bias",
        "lod_bias",
        model.get("lod_bias") || 1.0,
        0.001,
        5.0,
        true // settled-only (emit on change, not input)
      );
    }

    el.appendChild(section);

    // Helper: build a labeled float range row and wire it.
    // Returns the row element so callers can toggle visibility.
    function buildFloatRow(parent, labelText, traitName, initVal, minVal, maxVal, settledOnly) {
      const row = document.createElement("div");
      row.className = "cellier-app-row";

      const lbl = document.createElement("label");
      lbl.className = "cellier-app-label";
      lbl.textContent = labelText;

      const inp = document.createElement("input");
      inp.type = "range";
      inp.min = minVal;
      inp.max = maxVal;
      inp.step = "any";
      inp.value = initVal;

      const readout = document.createElement("span");
      readout.className = "cellier-app-readout";
      readout.textContent = Number(initVal).toFixed(3);

      if (!settledOnly) {
        inp.addEventListener("input", () => {
          if (guard) return;
          readout.textContent = Number(inp.value).toFixed(3);
          model.set(traitName, parseFloat(inp.value));
          model.save_changes();
        });
      }
      inp.addEventListener("change", () => {
        if (guard) return;
        readout.textContent = Number(inp.value).toFixed(3);
        model.set(traitName, parseFloat(inp.value));
        model.save_changes();
      });

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
      parent.appendChild(row);
      return row;
    }

    function applyModeVisibility(mode) {
      if (isoRow) {
        isoRow.style.display =
          mode === "iso" || mode === "smooth_iso" ? "" : "none";
      }
      if (attenuationRow) {
        attenuationRow.style.display = mode === "attenuated_mip" ? "" : "none";
      }
    }

    // ── AABB section ──────────────────────────────────────────────────────

    const aabbSection = document.createElement("div");
    aabbSection.className = "cellier-aabb";

    const aabbTitle = document.createElement("div");
    aabbTitle.className = "cellier-section-title";
    aabbTitle.textContent = "Bounding box";
    aabbSection.appendChild(aabbTitle);

    // enabled checkbox
    const enabledRow = document.createElement("div");
    enabledRow.className = "cellier-app-row";
    const enabledCheck = document.createElement("input");
    enabledCheck.type = "checkbox";
    enabledCheck.checked = model.get("aabb_enabled") || false;
    const enabledLabel = document.createElement("label");
    enabledLabel.className = "cellier-app-label cellier-aabb-check-label";
    enabledLabel.textContent = "Show";
    enabledRow.appendChild(enabledLabel);
    enabledRow.appendChild(enabledCheck);
    aabbSection.appendChild(enabledRow);

    enabledCheck.addEventListener("change", () => {
      if (guard) return;
      model.set("aabb_enabled", enabledCheck.checked);
      model.save_changes();
    });
    model.on("change:aabb_enabled", () => {
      guard = true;
      try { enabledCheck.checked = model.get("aabb_enabled") || false; }
      finally { guard = false; }
    });

    // line width
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
    lwInput.value = model.get("aabb_line_width") || 2.0;
    lwRow.appendChild(lwLabel);
    lwRow.appendChild(lwInput);
    aabbSection.appendChild(lwRow);

    lwInput.addEventListener("change", () => {
      if (guard) return;
      model.set("aabb_line_width", parseFloat(lwInput.value));
      model.save_changes();
    });
    model.on("change:aabb_line_width", () => {
      guard = true;
      try { lwInput.value = model.get("aabb_line_width") || 2.0; }
      finally { guard = false; }
    });

    // color
    const colorRow = document.createElement("div");
    colorRow.className = "cellier-app-row";
    const colorLabel = document.createElement("label");
    colorLabel.className = "cellier-app-label";
    colorLabel.textContent = "Color";
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.className = "cellier-aabb-color";
    colorInput.value = model.get("aabb_color") || "#ffffff";
    colorRow.appendChild(colorLabel);
    colorRow.appendChild(colorInput);
    aabbSection.appendChild(colorRow);

    colorInput.addEventListener("change", () => {
      if (guard) return;
      model.set("aabb_color", colorInput.value);
      model.save_changes();
    });
    model.on("change:aabb_color", () => {
      guard = true;
      try { colorInput.value = model.get("aabb_color") || "#ffffff"; }
      finally { guard = false; }
    });

    el.appendChild(aabbSection);
  } // end buildAppearance

  // ── Dataset info section ──────────────────────────────────────────────────

  const infoHtml = model.get("dataset_info") || "";
  if (infoHtml) {
    const infoSection = document.createElement("div");
    infoSection.className = "cellier-dataset-info";
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Dataset info";
    details.appendChild(summary);
    const body = document.createElement("div");
    body.innerHTML = infoHtml;
    details.appendChild(body);
    infoSection.appendChild(details);
    el.appendChild(infoSection);
  }
}

export default { render };
