const copy = document.querySelector("#copy");
if (navigator.clipboard?.writeText) {
  copy.hidden = false;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(
        document.querySelector("#commands").textContent,
      );
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Select commands to copy";
    }
  });
}

async function enableTrace() {
  const slider = document.querySelector("#input-x");
  const marker = document.querySelector("#point");
  const response = await fetch("./assets/drone-trace.json");
  if (!response.ok) throw new Error("Missing recorded samples");
  const { points, branches } = await response.json();
  const cells = ["region", "program", "network"].map((id) =>
    document.getElementById(id),
  );
  function update() {
    const point = points[slider.valueAsNumber];
    const x = (point.x >= 0 ? "+" : "") + point.x.toFixed(3);
    document.querySelector("#x-value").value = x;
    cells[0].textContent = "R" + point.region;
    document.querySelector("#branch-id").textContent = cells[0].textContent;
    document.querySelector("#active-source").textContent = branches[point.region];
    cells[1].textContent = point.thrust.toFixed(6);
    cells[2].textContent = point.network.toFixed(6);
    marker.setAttribute("transform", `translate(${point.px} ${point.py})`);
    slider.setAttribute(
      "aria-valuetext",
      `Altitude error ${x}, region ${point.region}, program output ${point.thrust}`,
    );
  }
  slider.max = points.length - 1;
  slider.addEventListener("input", update);
  update();
  marker.removeAttribute("hidden");
  slider.disabled = false;
}

enableTrace().catch(() => {
  document.querySelector("#trace-note").textContent =
    "Showing the saved sample. Reload to adjust the input.";
});
