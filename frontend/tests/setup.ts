import "@testing-library/jest-dom/vitest";

// jsdom does not implement matchMedia; model a dark system preference by default.
window.matchMedia = (query: string) => ({
  matches: query === "(prefers-color-scheme: dark)", media: query, onchange: null,
  addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {},
  dispatchEvent: () => true,
});
