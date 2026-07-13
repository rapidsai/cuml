function openTargetDropdown() {
  const target = document.getElementById(
    decodeURIComponent(window.location.hash.slice(1)),
  );
  const dropdown = target?.matches("details")
    ? target
    : target?.closest("details");

  if (dropdown) {
    dropdown.open = true;
  }
}

window.addEventListener("DOMContentLoaded", openTargetDropdown);
window.addEventListener("hashchange", openTargetDropdown);
