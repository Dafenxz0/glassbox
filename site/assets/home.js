const copy = document.getElementById("copy");
const commands = document.getElementById("commands");
if (navigator.clipboard?.writeText) {
  copy.hidden = false;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(commands.textContent);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Select commands to copy";
    }
  });
}
