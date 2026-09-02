// Dataset-info detail block ESM (split out of the former composite panel.js).
//
// Draws either a flat `rows` table or, when the widget was built from a
// DatasetInfo, the `sections` list: inline rows, nested <details> blocks, and
// matrix tables.  Every cell is written with textContent -- a value is data
// read off a store, never markup.

function makeTable(rows) {
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
  return table;
}

function makeMatrix(section) {
  const table = document.createElement("table");
  table.classList.add("cellier-matrix");

  const head = document.createElement("tr");
  head.appendChild(document.createElement("th")); // corner spacer
  for (const label of section.col_labels) {
    const th = document.createElement("th");
    th.textContent = label;
    head.appendChild(th);
  }
  table.appendChild(head);

  section.values.forEach((row, index) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = section.row_labels[index] ?? "";
    tr.appendChild(th);
    for (const value of row) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });

  return table;
}

function makeSubBlock(label, body, collapsed) {
  const details = document.createElement("details");
  details.open = !collapsed;
  const summary = document.createElement("summary");
  summary.textContent = label;
  details.appendChild(summary);
  details.appendChild(body);
  return details;
}

function render({ model, el }) {
  el.classList.add("cellier-dataset-info");

  function build() {
    el.replaceChildren();
    const sections = model.get("sections") || [];
    const rows = model.get("rows") || [];
    if (!sections.length && !rows.length) return;

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = model.get("title") || "Dataset info";
    details.appendChild(summary);

    if (!sections.length) {
      // No sectioned description: draw the flat row list as one table.
      details.appendChild(makeTable(rows));
      el.appendChild(details);
      return;
    }

    for (const section of sections) {
      if (section.kind === "matrix") {
        details.appendChild(
          makeSubBlock(section.label, makeMatrix(section), false),
        );
      } else if (section.label) {
        details.appendChild(
          makeSubBlock(
            section.label,
            makeTable(section.rows || []),
            section.collapsed,
          ),
        );
      } else {
        details.appendChild(makeTable(section.rows || []));
      }
    }

    el.appendChild(details);
  }

  build();
  model.on("change:rows", build);
  model.on("change:sections", build);
  model.on("change:title", build);
}

export default { render };
