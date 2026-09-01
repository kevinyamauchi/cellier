// Dataset-info detail block ESM (split out of the former composite panel.js).

function render({ model, el }) {
  el.classList.add("cellier-dataset-info");

  function build() {
    el.replaceChildren();
    const rows = model.get("rows") || [];
    if (!rows.length) return;

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = model.get("title") || "Dataset info";
    details.appendChild(summary);

    // textContent per cell: a value is data, never markup.
    const table = document.createElement("table");
    for (const [label, value] of rows) {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = label;
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(th);
      tr.appendChild(td);
      table.appendChild(tr);
    }
    details.appendChild(table);
    el.appendChild(details);
  }

  build();
  model.on("change:rows", build);
  model.on("change:title", build);
}

export default { render };
