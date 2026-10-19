// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

function openTargetDropdown() {
  let targetId;
  try {
    targetId = decodeURIComponent(window.location.hash.slice(1));
  } catch {
    return;
  }

  const target = document.getElementById(targetId);
  const dropdown = target?.matches("details")
    ? target
    : target?.closest("details");

  if (dropdown) {
    dropdown.open = true;
  }
}

window.addEventListener("DOMContentLoaded", openTargetDropdown);
window.addEventListener("hashchange", openTargetDropdown);
