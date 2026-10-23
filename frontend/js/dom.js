// DOM helpers shared by all views.
export const $ = (s) => document.querySelector(s);

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function setStatus(text) {
  $("#status").textContent = text;
}
