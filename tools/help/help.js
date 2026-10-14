"use strict";

const form = document.getElementById("search-form");
const input = document.getElementById("search");
const panel = document.getElementById("search-results");
const article = document.getElementById("article");
const matches = document.getElementById("matches");
const status = document.getElementById("search-status");

function closeSearch() {
  panel.hidden = true;
  article.hidden = false;
}

function search() {
  const query = input.value.trim().toLocaleLowerCase();
  if (!query) { closeSearch(); return; }
  const words = query.split(/\s+/);
  const results = (window.helpSearch || []).map(entry => {
    const title = (entry.page + " " + entry.title).toLocaleLowerCase();
    const body = entry.text.toLocaleLowerCase();
    return {entry, score: words.every(word => (title + " " + body).includes(word))
      ? 1 + words.filter(word => title.includes(word)).length * 3 + (title.includes(query) ? 5 : 0) : 0};
  }).filter(result => result.score).sort((a, b) => b.score - a.score);
  matches.replaceChildren();
  for (const {entry} of results.slice(0, 40)) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = entry.url;
    link.textContent = entry.page === entry.title ? entry.title : entry.page + " · " + entry.title;
    link.addEventListener("click", closeSearch);
    const excerpt = document.createElement("p");
    const offset = Math.max(0, entry.text.toLocaleLowerCase().indexOf(words[0]) - 70);
    excerpt.textContent = (offset ? "…" : "") + entry.text.slice(offset, offset + 220)
      + (entry.text.length > offset + 220 ? "…" : "");
    item.append(link, excerpt);
    matches.append(item);
  }
  status.textContent = results.length ? `${results.length} matching sections${results.length > 40 ? " · showing the first 40" : ""}.`
    : "No matching sections. Try a control name or a shorter phrase.";
  panel.hidden = false;
  article.hidden = true;
}

form.addEventListener("submit", event => { event.preventDefault(); search(); });
input.addEventListener("input", search);
document.getElementById("close-search").addEventListener("click", () => { closeSearch(); input.focus(); });
document.addEventListener("keydown", event => { if (event.key === "Escape") closeSearch(); });
document.querySelectorAll("nav a").forEach(link => link.addEventListener("click", closeSearch));
window.addEventListener("hashchange", closeSearch);
if (window.location.hash === "#search") input.focus();
