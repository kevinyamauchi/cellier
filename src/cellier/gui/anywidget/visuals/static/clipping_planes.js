// Clipping planes ESM: one row per entry of `rows`.  The Python side syncs
// `rows` (what to draw) and `presets` (the normal presets); an action is
// reported by setting `edit`, and a refused one comes back as `error` with
// `rows` unchanged.
//
// A row is built once and then updated in place, so a slider is not
// destroyed while it is dragged.  Rows are only added or removed when the
// number of planes changes.

function render({ model, el }) {
  el.classList.add("cellier-clipping-planes");
  let serial = 0;

  const heading = document.createElement("div");
  heading.className = "cellier-clipping-planes-title";
  el.appendChild(heading);

  const body = document.createElement("div");
  body.className = "cellier-clipping-planes-rows";
  el.appendChild(body);

  const add = document.createElement("button");
  add.className = "cellier-clipping-planes-add";
  add.textContent = "Add plane";
  add.title = "Add a plane across the last axis, through the middle.";
  add.dataset.role = "add";
  el.appendChild(add);

  const error = document.createElement("div");
  error.className = "cellier-clipping-planes-error";
  el.appendChild(error);

  function send(action, index, value) {
    serial += 1;
    model.set("edit", { action, index, value, serial });
    model.save_changes();
  }

  add.addEventListener("click", () => send("add", null, null));

  // One entry per built row: { root, apply(row) }.
  const built = [];

  function build(index) {
    const root = document.createElement("div");
    root.className = "cellier-clipping-planes-row";
    root.dataset.role = "row";
    root.dataset.index = String(index);

    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.title = "Whether this plane clips. It stays in the list.";
    enabled.dataset.role = "enabled";
    enabled.addEventListener("change", () => send("enabled", index, enabled.checked));

    const preset = document.createElement("select");
    preset.title = "The data axis the plane's normal points along.";
    preset.dataset.role = "preset";
    for (const name of model.get("presets")) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      preset.appendChild(option);
    }
    preset.addEventListener("change", () => send("preset", index, preset.value));

    const normal = document.createElement("input");
    normal.type = "text";
    normal.className = "cellier-clipping-planes-normal";
    normal.title = "The normal, one number per data axis. It points to the side that is kept.";
    normal.dataset.role = "normal";
    normal.addEventListener("change", () => send("normal", index, normal.value));

    const flip = document.createElement("button");
    flip.textContent = "Flip";
    flip.title = "Keep the other side of the plane.";
    flip.dataset.role = "flip";
    flip.addEventListener("click", () => send("flip", index, null));

    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.dataset.role = "remove";
    remove.addEventListener("click", () => send("remove", index, null));

    const spacer = document.createElement("span");

    const position = document.createElement("input");
    position.type = "range";
    position.step = "any";
    position.className = "cellier-clipping-planes-position";
    position.title = "Where the plane sits along its normal. Data units.";
    position.dataset.role = "position";
    let dragging = false;
    position.addEventListener("input", () => {
      dragging = true;
      value.textContent = Number(position.value).toFixed(2);
      send("position", index, Number(position.value));
    });
    position.addEventListener("change", () => { dragging = false; });

    const value = document.createElement("span");
    value.className = "cellier-clipping-planes-value";
    value.dataset.role = "value";

    root.append(enabled, preset, normal, flip, remove, spacer, position, value);

    function apply(row) {
      enabled.checked = Boolean(row.enabled);
      preset.value = row.preset;
      if (document.activeElement !== normal) normal.value = row.normal_text;
      position.min = String(row.low);
      position.max = String(row.high);
      // While the thumb is held the element already shows the newest value;
      // writing an older echo back would make it jump.
      if (!dragging) position.value = String(row.position);
      value.textContent = Number(row.position).toFixed(2);
    }
    return { root, apply };
  }

  function draw() {
    const rows = model.get("rows");
    while (built.length > rows.length) {
      built.pop().root.remove();
    }
    while (built.length < rows.length) {
      const entry = build(built.length);
      built.push(entry);
      body.appendChild(entry.root);
    }
    rows.forEach((row, index) => built[index].apply(row));
  }

  function drawTitle() {
    heading.textContent = model.get("title");
    heading.style.display = model.get("title") ? "" : "none";
  }

  function drawError() {
    error.textContent = model.get("error");
  }

  model.on("change:rows", draw);
  model.on("change:title", drawTitle);
  model.on("change:error", drawError);
  drawTitle();
  draw();
  drawError();
}

export default { render };
