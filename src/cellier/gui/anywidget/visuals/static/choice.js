// Select control ESM -- one asset shared by every Literal appearance field.
//
// Field-agnostic: generic `value` / `label` / `choices` traits. The options
// come from the model's own Literal annotation on the Python side, so this
// file never names a mode and a mode added to a model appears here with no
// change.

function render({ model, el }) {
  el.classList.add("cellier-choice");

  let guard = false;

  const row = document.createElement("div");
  row.className = "cellier-app-row";

  const label = document.createElement("label");
  label.className = "cellier-app-label";
  label.textContent = model.get("label") || "";

  const select = document.createElement("select");
  select.className = "cellier-choice-select";

  function applyChoices() {
    const choices = model.get("choices") || [];
    select.replaceChildren();
    for (const choice of choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      select.appendChild(option);
    }
    applyValue();
  }

  function applyValue() {
    const value = model.get("value");
    // Assigning an absent value would blank the select; leave it as it is,
    // matching QtChoice._apply.
    if (Array.from(select.options).some((o) => o.value === value)) {
      select.value = value;
    }
  }

  applyChoices();

  select.addEventListener("change", () => {
    if (guard) return;
    model.set("value", select.value);
    model.save_changes();
  });

  model.on("change:value", () => {
    guard = true;
    try {
      applyValue();
    } finally {
      guard = false;
    }
  });
  model.on("change:choices", applyChoices);
  model.on("change:label", () => {
    label.textContent = model.get("label") || "";
  });

  row.appendChild(label);
  row.appendChild(select);
  el.appendChild(row);
}

export default { render };
