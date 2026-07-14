// Dataset-info detail block ESM (split out of the former composite panel.js).

function render({ model, el }) {
  el.classList.add("cellier-dataset-info");

  function build() {
    el.replaceChildren();
    const infoHtml = model.get("dataset_info") || "";
    if (!infoHtml) return;

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Dataset info";
    details.appendChild(summary);
    const body = document.createElement("div");
    body.innerHTML = infoHtml;
    details.appendChild(body);
    el.appendChild(details);
  }

  build();
  model.on("change:dataset_info", build);
}

export default { render };
